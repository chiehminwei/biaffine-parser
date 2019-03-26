# -*- coding: utf-8 -*-

from collections import Counter

import regex
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer
import numpy as np


class Vocab(object):
    PAD = '<PAD>'

    def __init__(self, words, chars, rels):
        self.pad_index = 0

        self.words = [self.PAD] + sorted(words)
        self.chars = [self.PAD] + sorted(chars)
        self.rels = sorted(rels)

        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.char_dict = {char: i for i, char in enumerate(self.chars)}
        self.rel_dict = {rel: i for i, rel in enumerate(self.rels)}

        # ids of punctuation that appear in words
        self.puncts = set(sorted(i for word, i in self.word_dict.items()
                                if regex.match(r'\p{P}+$', word)))

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

    def rel2id(self, sequence):
        return torch.tensor([self.rel_dict.get(rel, 0)
                             for rel in sequence])

    def id2rel(self, ids):
        return [self.rels[i] for i in ids]

    def extend(self, words):
        self.words += sorted(set(words).difference(self.word_dict))
        self.chars += sorted(set(''.join(words)).difference(self.char_dict))
        self.word_dict = {w: i for i, w in enumerate(self.words)}
        self.char_dict = {c: i for i, c in enumerate(self.chars)}
        self.puncts = set(sorted(i for word, i in self.word_dict.items()
                                if regex.match(r'\p{P}+$', word)))
        self.n_words = len(self.words)
        self.n_chars = len(self.chars)

    def numericalize(self, corpus):
        words_numerical = []
        arcs_numerical = []
        rels_numerical = []
        token_start_mask = []
        attention_mask = []
        flag = False
        for words, arcs, rels in zip(corpus.words, corpus.heads, corpus.rels):
            sentence_token_ids = []
            sentence_arc_ids = []
            sentence_rel_ids = []
            token_starts = []
            attentions = []
            words = ['[CLS]'] + words + ['[SEP]']
            arcs = [0] + arcs + [0]
            rels = ['<ROOT>'] + rels + ['<ROOT>']
            for word, arc, rel in zip(words, arcs, rels):
                if word == '<ROOT>':
                    continue
                else:
                    if word == '`':
                        word = "'"
                    if word == '``':
                        word = '"'
                    if word == "''":
                        word = '"'
                    if word == "non-``":
                        word = 'non-"'
                    word = word.replace("`", "'")
                    word = word.replace('“', '"')
                    word = word.replace('”', '"')
                    word = word.replace("’", "'")
                    word = word.replace("…", "...")
                    word = word.replace("–", "-")
                    

                    tokens = self.tokenizer.tokenize(word)
                    ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    if regex.match(r'\p{P}+$', word):
                        for token_id in ids:
                            self.puncts.add(token_id)

                    if '[UNK]' in tokens:
                        print(word)
                        # print(tokens)
                        flag = True
                        # raise RuntimeError('Illegal character found in corpus.')

                sentence_token_ids.extend(ids)
                sentence_arc_ids.extend([arc] * len(tokens))
                sentence_rel_ids.extend([self.rel_dict.get(rel, 0)] * len(tokens))
                token_starts.extend([1] + [0] * (len(tokens) - 1))
                attentions.extend([1] * len(tokens))
            words_numerical.append(torch.tensor(sentence_token_ids))
            arcs_numerical.append(torch.tensor(sentence_arc_ids))
            rels_numerical.append(torch.tensor(sentence_rel_ids))
            token_start_mask.append(torch.ByteTensor(token_starts))
            attention_mask.append(torch.ByteTensor(attentions))

        if flag: raise RuntimeError('Illegal character found in corpus.')
        return words_numerical, attention_mask, token_start_mask, arcs_numerical, rels_numerical

    @classmethod
    def from_corpus(cls, corpus, min_freq=1):
        words = Counter(word for seq in corpus.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        chars = list({char for seq in corpus.words for char in ''.join(seq)})
        rels = list({rel for seq in corpus.rels for rel in seq})
        vocab = cls(words, chars, rels)

        return vocab
