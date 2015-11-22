from __future__ import division
from collections import Counter, defaultdict
from random import shuffle
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class GloVeModel():
    def __init__(self, embedding_size, context_size, max_vocab_size=None, min_occurrences=1,
                 scaling_factor=3/4, cooccurrence_cap=100, batch_size=512, learning_rate=0.05,
                 session=None):
        self.embedding_size = embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        elif isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")
        self.max_vocab_size = max_vocab_size
        self.min_occurrences = min_occurrences
        self.scaling_factor = scaling_factor
        self.cooccurrence_cap = cooccurrence_cap
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit_to_corpus(self, corpus):
        self._fit_to_corpus(corpus, self.max_vocab_size, self.min_occurrences,
                            self.left_context, self.right_context)
        self.build_graph()

    def _fit_to_corpus(self, corpus, vocab_size, min_occurrences, left_size, right_size):
        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)
        for region in corpus:
            word_counts.update(region)
            for left_context, word, right_context in context_windows(region, left_size, right_size):
                for i, context_word in enumerate(left_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(right_context):
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
        self.words = [word for word, count in word_counts.most_common(vocab_size)
                      if count >= min_occurrences]
        self.word_index = {word: i for i, word in enumerate(self.words)}
        self.cooccurrence_matrix = {
            (self.word_index[words[0]], self.word_index[words[1]]): count
            for words, count in cooccurrence_counts.items()
            if words[0] in self.word_index and words[1] in self.word_index}

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default(), self.graph.device(device_for_node):
            count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32,
                                    name='max_cooccurrence_count')
            scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32,
                                         name="scaling_factor")

            self.focal_input = tf.placeholder(tf.int32, shape=[self.batch_size], name="focal_words")
            self.context_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="context_words")
            self.cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                     name="cooccurrence_count")

            focal_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                name="focal_embeddings")
            context_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                name="context_embeddings")

            focal_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                       name='focal_biases')
            context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                         name="context_biases")

            focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.focal_input)
            context_embedding = tf.nn.embedding_lookup([context_embeddings], self.context_input)
            focal_bias = tf.nn.embedding_lookup([focal_biases], self.focal_input)
            context_bias = tf.nn.embedding_lookup([context_biases], self.context_input)

            weighting_factor = tf.minimum(
                1.0,
                tf.pow(
                    tf.div(self.cooccurrence_count, count_max),
                    scaling_factor))

            embedding_product = tf.reduce_sum(tf.mul(focal_embedding, context_embedding), 1)

            log_cooccurrences = tf.log(tf.to_float(self.cooccurrence_count))

            distance_expr = tf.square(tf.add_n([
                embedding_product,
                focal_bias,
                context_bias,
                tf.neg(log_cooccurrences)]))

            single_losses = tf.mul(weighting_factor, distance_expr)
            self.total_loss = tf.reduce_sum(single_losses)
            tf.scalar_summary("GloVe loss", self.total_loss)
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(
                self.total_loss)
            self.summary = tf.merge_all_summaries()

            self.combined_embeddings = tf.add(focal_embeddings, context_embeddings,
                                              name="combined_embeddings")

    def train(self, num_epochs, log_dir=None, report_interval=10000, tsne_output_interval=5):
        batches = self.prepare_batches()
        total_steps = 0
        with tf.Session(graph=self.graph) as session:
            summary_writer = tf.train.SummaryWriter(log_dir, graph_def=session.graph_def)
            tf.initialize_all_variables().run()
            for epoch in range(num_epochs):
                shuffle(batches)
                for batch_index, batch in enumerate(batches):
                    i_s, j_s, counts = batch
                    if len(counts) != self.batch_size:
                        continue
                    feed_dict = {
                        self.focal_input: i_s,
                        self.context_input: j_s,
                        self.cooccurrence_count: counts}
                    _, total_loss_ = session.run([self.optimizer, self.total_loss],
                                                 feed_dict=feed_dict)
                    if log_dir is not None and (total_steps + 1) % report_interval == 0:
                        summary_str = session.run(self.summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, total_steps)
                    total_steps += 1
                if (epoch + 1) % tsne_output_interval == 0:
                    current_embeddings = self.combined_embeddings.eval()
                    output_tsne(current_embeddings, self.words, "epoch{:03d}.png".format(epoch + 1))
            final_embeddings = self.combined_embeddings.eval()
            summary_writer.close()
        return final_embeddings

    def prepare_batches(self):
        cooccurrences = [(pos[0], pos[1], count) for pos, count in self.cooccurrence_matrix.items()]
        i_indices, j_indices, counts = zip(*cooccurrences)
        return list(batchify(self.batch_size, i_indices, j_indices, counts))

    @property
    def vocab_size(self):
        return len(self.words)


def context_windows(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = window(region, start_index, i - 1)
        right_context = window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def window(region, start_index, end_index):
    """
    Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
    return selected_tokens


def device_for_node(n):
    if n.type == "MatMul":
        return "/gpu:0"
    else:
        return "/cpu:0"


def batchify(batch_size, *sequences):
    for i in xrange(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i+batch_size] for sequence in sequences)


def output_tsne(embeddings, words, filename, size=(100, 100), plot_only=1000):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
    labels = words[:plot_only]
    plot_with_labels(low_dim_embs, labels, filename, size)


def plot_with_labels(low_dim_embs, labels, filename='tsne.png', size=(100, 100)):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=size)  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    figure.savefig(filename)
    plt.close(figure)
