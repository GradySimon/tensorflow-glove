from __future__ import division
from collections import Counter, defaultdict
import re
import os
import nltk

NULL_WORD = "<null>"


class Corpus(object):
    def __init__(self, **kwargs):
        """
        `size`: How many words on each side to select for each window. Specifying
        `size` gives symmetric context windows and is equivalent to setting
        `left_size` and `right_size` to the same value.
        """
        if 'size' in kwargs:
            self.left_size = self.right_size = kwargs['size']
        elif 'left_size' in kwargs or 'right_size' in kwargs:
            self.left_size = kwargs.get('left_size', 0)
            self.right_size = kwargs.get('right_size', 0)
        else:
            raise KeyError("At least one of `size`, `left_size`, and `right_size` must be given")
        self._words = None
        self._word_index = None
        self._cooccurrence_matrix = None

    def tokenized_regions(self):
        """
        Returns an iterable of all tokenized regions of text from the corpus.
        """
        return map(self.tokenize, self.extract_regions())

    def fit(self, vocab_size=None, min_occurrences=1):
        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)
        for region in self.tokenized_regions():
            word_counts.update(region)
            for left_context, word, right_context in self.region_context_windows(region):
                for i, context_word in enumerate(left_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(right_context):
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
        self._words = [word for word, count in word_counts.most_common(vocab_size)
                       if count >= min_occurrences]
        self._word_index = {word: i for i, word in enumerate(self._words)}
        word_set = set(self._words)
        self._cooccurrence_matrix = {
            (self._word_index[words[0]], self._word_index[words[1]]): count
            for words, count in cooccurrence_counts.items()
            if words[0] in word_set and words[1] in word_set
        }

    def is_fit(self):
        """
        Returns a boolean for whether or not the Corpus object has been fit to
        the text yet.
        """
        return self._words is not None

    def extract_regions(self):
        """
        Returns an iterable of all regions of text (strings) in the corpus that
        should each be considered one contiguous unit. Messages, comments,
        reviews, etc.
        """
        raise NotImplementedError()

    @staticmethod
    def tokenize(string):
        """
        Takes strings that come from `extract_regions` and returns the tokens
        from that string.
        """
        raise NotImplementedError()

    @staticmethod
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

    def region_context_windows(self, region):
        for i, word in enumerate(region):
            start_index = i - self.left_size
            end_index = i + self.right_size
            left_context = self.window(region, start_index, i - 1)
            right_context = self.window(region, i + 1, end_index)
            yield (left_context, word, right_context)

    @property
    def words(self):
        if not self.is_fit():
            self.fit()
        return self._words

    @property
    def word_index(self):
        if not self.is_fit():
            self.fit()
        return self._word_index

    @property
    def cooccurrence_matrix(self):
        if self._cooccurrence_matrix is None:
            self.fit()
        return self._cooccurrence_matrix


class RedditCorpus(Corpus):
    def __init__(self, path, **kwargs):
        """
        If `path` is a file, it will be treated as the only file in the corpus.
        If it's a directory, every file not starting with a dot (".") will be
        considered to be in the corpus. See the constructor for `Corpus` for
        keyword args.
        """
        super(RedditCorpus, self).__init__(**kwargs)
        if os.path.isdir(path):
            file_names = filter(lambda p: not p.startswith('.'), os.listdir(path))
            self.file_paths = [os.path.join(path, name) for name in file_names]
        else:
            self.file_paths = [path]

    def extract_regions(self):
        # This is not exactly a rock-solid way to get the body, but it's ~2x as
        # fast as json parsing each line
        body_snatcher = re.compile(r"\{.*?(?<!\\)\"body(?<!\\)\":(?<!\\)\"(.*?)(?<!\\)\".*}")
        for file_path in self.file_paths:
            with open(file_path) as file_:
                for line in file_:
                    match = body_snatcher.match(line)
                    if match:
                        body = match.group(1)
                        if not body == '[deleted]':
                            yield body

    @staticmethod
    def tokenize(string):
        return nltk.wordpunct_tokenize(string.lower())
