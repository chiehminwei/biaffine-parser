# -*- coding: utf-8 -*-

from collections import Counter, namedtuple

import torch


Sentence = namedtuple(typename='Sentence',
                      field_names=['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS',
                                   'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC'],
                      defaults=[None]*10)


class Corpus(object):
    ROOT = '<ROOT>'

    def __init__(self, sentences):
        super(Corpus, self).__init__()

        self.sentences = sentences

    def __repr__(self):
        return '\n'.join(
            '\n'.join('\t'.join(map(str, i)) for i in zip(*sentence)) + '\n'
            for sentence in self
        )

    def __getitem__(self, index):
        return self.sentences[index]

    @property
    def words(self):
        return Counter(word.lower() for seq in self.word_seqs
                       for word in seq[1:])

    @property
    def labels(self):
        return Counter(label for seq in self.label_seqs
                       for label in seq[1:])

    @property
    def word_seqs(self):
        return [[self.ROOT] + [word for word in sentence.FORM]
                for sentence in self.sentences]

    @property
    def head_seqs(self):
        return [[0] + list(map(int, sentence.HEAD))
                for sentence in self.sentences]

    @property
    def label_seqs(self):
        return [[self.ROOT] + list(sentence.DEPREL)
                for sentence in self.sentences]

    @head_seqs.setter
    def head_seqs(self, sequences):
        self.sentences = [sentence._replace(HEAD=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @label_seqs.setter
    def label_seqs(self, sequences):
        self.sentences = [sentence._replace(DEPREL=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @classmethod
    def load(cls, fname, field_names=Sentence._fields):
        start, sentences = 0, []
        with open(fname, 'r') as f:
            lines = [line for line in f]
        for i, line in enumerate(lines):
            if len(line) <= 1:
                cols = zip(*[l.split() for l in lines[start:i]])
                fields = dict(zip(field_names, cols))
                sentences.append(Sentence(**fields))
                start = i + 1
        corpus = cls(sentences)

        return corpus

    def dump(self, fname):
        with open(fname, 'w') as f:
            f.write(f"{self}\n")


class Embedding(object):

    def __init__(self, words, vectors):
        super(Embedding, self).__init__()

        self.words = words
        self.vectors = vectors
        self.pretrained = {w: v for w, v in zip(words, vectors)}

    def __contains__(self, word):
        return word in self.pretrained

    def __getitem__(self, word):
        return torch.tensor(self.pretrained[word])

    @property
    def dim(self):
        return len(self.vectors[0])

    @classmethod
    def load(cls, fname):
        with open(fname, 'r') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines]
        reprs = [(s[0], list(map(float, s[1:]))) for s in splits]
        words, vectors = map(list, zip(*reprs))
        embedding = cls(words, vectors)

        return embedding
