from __future__ import division
from collections import Counter, defaultdict

class GloVeModel():
    """docstring for GloVeModel"""
    def __init__(self, embedding_size, scaling_factor=3/4, cooccurrence_cap=100, batch_size=512,
                 learning_rate=0.05):
        self.embedding_size = embedding_size
        self.scaling_factor = scaling_factor
        self.cooccurrence_cap = cooccurrence_cap
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def process_corpus(self, corpus, vocab_size=None, min_occurrences=1, **kwargs):
        if 'context' in kwargs:
            left_context = right_context = kwargs['context']
        elif 'left_context' in kwargs or 'right_context' in kwargs:
            left_context = kwargs.get('left_context', 0)
            right_context = kwargs.get('right_context', 0)
        else:
            raise KeyError(
                "At least one of `context`, `left_context`, and `right_context` must be given")
        self.words, self.word_index, self.cooccurrence_matrix = build_cooccurrence_matrix(
            corpus, vocab_size, min_occurrences, left_context, right_context)

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
        words = [word for word, count in word_counts.most_common(vocab_size) if count >= min_occurrences]
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
