from __future__ import division
from collections import Counter, defaultdict
import tensorflow as tf


class GloVeModel():
    def __init__(self, embedding_size, context_size, vocab_size=None, min_occurrences=1,
                 scaling_factor=3/4, cooccurrence_cap=100, batch_size=512, learning_rate=0.05):
        self.embedding_size = embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        elif isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")
        self.vocab_size = vocab_size
        self.min_occurrences = min_occurrences
        self.scaling_factor = scaling_factor
        self.cooccurrence_cap = cooccurrence_cap
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self, corpus):
        build_cooccurrence_matrix(corpus, self.vocab_size, self.min_occurrences, self.left_context,
                                  self.right_context)

    def build_graph(self):
        with graph.as_default():
            with graph.device(device_for_node):
                count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32)
                scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32)

                focal_input = tf.placeholder(tf.int32, shape=[self.batch_size])
                context_input = tf.placeholder(tf.int32, shape=[self.batch_size])
                cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size])

                focal_embeddings = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0))
                context_embeddings = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0))

                focal_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0))
                context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0))

                focal_embedding = tf.nn.embedding_lookup([focal_embeddings], focal_input)
                context_embedding = tf.nn.embedding_lookup([context_embeddings], context_input)
                focal_bias = tf.nn.embedding_lookup([focal_biases], focal_input)
                context_bias = tf.nn.embedding_lookup([context_biases], context_input)

                weighting_factor = tf.minimum(
                    1.0,
                    tf.pow(
                        tf.div(cooccurrence_count, count_max),
                        scaling_factor))

                embedding_product = tf.reduce_sum(tf.mul(focal_embedding, context_embedding), 1)

                log_cooccurrences = tf.log(tf.to_float(cooccurrence_count))

                distance_expr = tf.square(tf.add_n([
                    embedding_product,
                    focal_bias,
                    context_bias,
                    tf.neg(log_cooccurrences)]))

                single_losses = tf.mul(weighting_factor, distance_expr)
                total_loss = tf.reduce_sum(single_losses)
                optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(total_loss)

                combined_embeddings = tf.add(focal_embeddings, context_embeddings)


def build_cooccurrence_matrix(corpus, vocab_size, min_occurrences, left_size, right_size):
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
        words = [word for word, count in word_counts.most_common(vocab_size)
                 if count >= min_occurrences]
        word_index = {word: i for i, word in enumerate(words)}
        word_set = set(words)
        cooccurrence_matrix = {
            (word_index[words[0]], word_index[words[1]]): count
            for words, count in cooccurrence_counts.items()
            if words[0] in word_set and words[1] in word_set
        }
        return words, word_index, cooccurrence_matrix


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
