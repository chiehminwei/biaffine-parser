# -*- coding: utf-8 -*-

from collections import Counter

import regex
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer
import numpy as np

class Vocab(object):
    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self, words, chars, rels):
        self.pad_index = 0
        self.unk_index = 1

        self.words = [self.PAD, self.UNK] + sorted(words)
        self.chars = [self.PAD, self.UNK] + sorted(chars)
        self.rels = sorted(rels)

        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.char_dict = {char: i for i, char in enumerate(self.chars)}
        self.rel_dict = {rel: i for i, rel in enumerate(self.rels)}

        # ids of punctuation that appear in words
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if regex.match(r'\p{P}+$', word))

        self.n_words = len(self.words)
        self.n_chars = len(self.chars)
        self.n_rels = len(self.rels)
        self.n_train_words = self.n_words

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    def __repr__(self):
        info = f"{self.__class__.__name__}(\n"
        info += f"  num of words: {self.n_words}\n"
        info += f"  num of chars: {self.n_chars}\n"
        info += f"  num of rels: {self.n_rels}\n"
        info += f")"

        return info

    def word2id(self, sequence):
        return torch.tensor([self.word_dict.get(word.lower(), self.unk_index)
                             for word in sequence])

    def char2id(self, sequence, max_length=20):
        char_ids = torch.zeros(len(sequence), max_length, dtype=torch.long)
        for i, word in enumerate(sequence):
            ids = torch.tensor([self.char_dict.get(c, self.unk_index)
                                for c in word[:max_length]])
            char_ids[i, :len(ids)] = ids

        return char_ids

    def rel2id(self, sequence):
        return torch.tensor([self.rel_dict.get(rel, 0)
                             for rel in sequence])

    def id2rel(self, ids):
        return [self.rels[i] for i in ids]

    def read_embeddings(self, embed, unk=None):
        words = embed.words
        # if the UNK token has existed in pretrained vocab,
        # then replace it with a self-defined one
        if unk in embed:
            words[words.index(unk)] = self.UNK

        self.extend(words)
        self.embeddings = torch.zeros(self.n_words, embed.dim)

        for i, word in enumerate(self.words):
            if word in embed:
                self.embeddings[i] = embed[word]
        self.embeddings /= torch.std(self.embeddings)

    def extend(self, words):
        self.words += sorted(set(words).difference(self.word_dict))
        self.chars += sorted(set(''.join(words)).difference(self.char_dict))
        self.word_dict = {w: i for i, w in enumerate(self.words)}
        self.char_dict = {c: i for i, c in enumerate(self.chars)}
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if regex.match(r'\p{P}+$', word))
        self.n_words = len(self.words)
        self.n_chars = len(self.chars)

    def numericalize(self, corpus):
        words_numerical = []
        arcs_numerical = []
        rels_numerical = []
        token_start_mask = []
        for words, arcs, rels in zip(corpus.words, corpus.heads, corpus.rels):
            sentence_token_ids = []
            sentence_arc_ids = []
            sentence_rel_ids = []
            token_starts = []
            words = ['[CLS]'] + words + ['[SEP]']
            arcs = [0] + arcs + [0]
            rels = ['PAD'] + rels + ['PAD']
            for word, arc, rel in zip(words, arcs, rels):
                if word == '<ROOT>':
                    tokens = ['<ROOT>']
                    ids = [0]
                else:
                    tokens = self.tokenizer.tokenize(word)
                    ids = self.tokenizer.convert_tokens_to_ids(tokens)
                sentence_token_ids.extend(ids)
                sentence_arc_ids.extend([arc] * len(tokens))
                sentence_rel_ids.extend([self.rel_dict.get(rel, 0)] * len(tokens))
                token_starts.extend([1] + [0] * (len(tokens) - 1))
            words_numerical.append(sentence_token_ids)
            arcs_numerical.append(sentence_arc_ids)
            rels_numerical.append(sentence_rel_ids)
            token_start_mask.append(token_starts)
        return words_numerical, token_start_mask, arcs_numerical, rels_numerical


    def yeet(self, corpus):
        words = [self.word2id(seq) for seq in corpus.words]
        chars = [self.char2id(seq) for seq in corpus.words]
        arcs = [torch.tensor(seq) for seq in corpus.heads]
        rels = [self.rel2id(seq) for seq in corpus.rels]

        return words, chars, arcs, rels

    @classmethod
    def from_corpus(cls, corpus, min_freq=1):
        words = Counter(word.lower() for seq in corpus.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        chars = list({char for seq in corpus.words for char in ''.join(seq)})
        rels = list({rel for seq in corpus.rels for rel in seq})
        vocab = cls(words, chars, rels)

        return vocab
