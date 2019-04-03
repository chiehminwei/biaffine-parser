# -*- coding: utf-8 -*-

from collections import Counter

import regex
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer
import numpy as np
import unicodedata


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

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

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

    def numericalize(self, corpus, save_name=None):
        words_numerical = []
        arcs_numerical = []
        rels_numerical = []
        token_start_mask = []
        attention_mask = []
        flag = False
        error_flag = False
        offending_set = set()
        symbol_set = set()
        empty_words = set()
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
                # skip <ROOT>
                if word == '<ROOT>':
                    continue
                
                # take care of some idiosyncracies
                if word == '`':
                    word = "'"
                if word == '``':
                    word = '"'
                if word == "''":
                    word = '"'
                if word == "non-``":
                    word = 'non-"'
                word = word.replace('“', '"')
                word = word.replace('”', '"')
                word = word.replace("`", "'")
                word = word.replace("’", "'")
                word = word.replace("‘", "'")
                word = word.replace("'", "'")
                word = word.replace("´", "'")
                word = word.replace("…", "...")
                word = word.replace("–", "-")
                word = word.replace('—', '-')


                tokens = self.tokenizer.tokenize(word)                
                if tokens:
                    ids = self.tokenizer.convert_tokens_to_ids(tokens)

                    # take care of punctuation
                    if regex.match(r'\p{P}+$', word):
                        for token_id in ids:
                            self.puncts.add(token_id)

                    # log any unknown words
                    if '[UNK]' in tokens:
                        # print('words: ', words)
                        # print('offending word: ', word)
                        # print('offending chars: ')
                        for offending_char in word:
                            token = self.tokenizer.tokenize(offending_char)
                            if '[UNK]' in token:
                                if unicodedata.category(offending_char) != 'So':
                                    offending_set.add(offending_char)
                                else:
                                    symbol_set.add(offending_char)
                        flag = True    

                    # main thing to do
                    sentence_token_ids.extend(ids)
                    sentence_arc_ids.extend([arc] * len(tokens))
                    sentence_rel_ids.extend([self.rel_dict.get(rel, 0)] * len(tokens))
                    token_starts.extend([1] + [0] * (len(tokens) - 1))
                    attentions.extend([1] * len(tokens))

                # take care of empty tokens
                else:
                    # print('\noffending word: ', word)
                    # print('empty words: ', ' '.join(words))
                    empty_words.add(word)
                    error_flag = True
                    continue
                
                # error checking for lengths
                len_sentence_token_ids = len(sentence_token_ids)
                len_sentence_arc_ids = len(sentence_arc_ids)
                len_sentence_rel_ids = len(sentence_rel_ids)
                len_token_starts = len(token_starts)
                len_attentions = len(attentions)
                if not (len_sentence_token_ids == len_sentence_arc_ids == len_sentence_rel_ids == len_token_starts == len_attentions):
                    print(words)
                    print(arcs)
                    print(rels)
                    print('len_sentence_token_ids: ', len_sentence_token_ids)
                    print('len_sentence_arc_ids', len_sentence_arc_ids)
                    print('len_sentence_rel_ids', len_sentence_rel_ids)
                    print('len_token_starts', len_token_starts)
                    print('len_attentions', len_attentions)
                    raise RuntimeError("Lengths don't match up.")


            words_numerical.append(torch.tensor(sentence_token_ids))
            arcs_numerical.append(torch.tensor(sentence_arc_ids))
            rels_numerical.append(torch.tensor(sentence_rel_ids))
            token_start_mask.append(torch.ByteTensor(token_starts))
            attention_mask.append(torch.ByteTensor(attentions))

        if flag: 
            print('WARNING: The following non-symbol characters are unknown to BERT:')
            print(offending_set)
            print('WARNING: The following symbol characters are unknown to BERT:')            
            print(symbol_set)
            
        if error_flag:
            print('WARNING: The following characters are empty after going through tokenizer:')
            print(empty_words)
            # raise RuntimeError('Some tokens are empty.')
        if save_name:
            torch.save((words_numerical, attention_mask, token_start_mask, arcs_numerical, rels_numerical), save_name)
        
        return words_numerical, attention_mask, token_start_mask, arcs_numerical, rels_numerical

    def numericalize_sentences(self, sentences):
        words_numerical = []
        token_start_mask = []
        attention_mask = []
        flag = False
        offending_set = set()
        for sentence in sentences:
            sentence_token_ids = []
            token_starts = []
            attentions = []
            sentence = ['[CLS]'] + sentence + ['[SEP]']
            for word in sentence:
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
                    word = word.replace('“', '"')
                    word = word.replace('”', '"')

                    word = word.replace("`", "'")
                    word = word.replace("’", "'")
                    word = word.replace("‘", "'")
                    word = word.replace("'", "'")
                    word = word.replace("´", "'")

                    word = word.replace("…", "...")

                    word = word.replace("–", "-")
                    word = word.replace('—', '-')


                    tokens = self.tokenizer.tokenize(word)
                    ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    if regex.match(r'\p{P}+$', word):
                        for token_id in ids:
                            self.puncts.add(token_id)

                    if '[UNK]' in tokens:
                        for offending_char in word:
                            token = self.tokenizer.tokenize(offending_char)
                            if '[UNK]' in token:
                                offending_set.add(offending_char)
                        flag = True
                        
                sentence_token_ids.extend(ids)
                token_starts.extend([1] + [0] * (len(tokens) - 1))
                attentions.extend([1] * len(tokens))
                
            words_numerical.append(torch.tensor(sentence_token_ids))
            attention_mask.append(torch.ByteTensor(attentions))
            token_start_mask.append(torch.ByteTensor(token_starts))
            
        if flag: 
            print('WARNING: The following characters are unknown to BERT:')
            print(offending_set)

        return words_numerical, attention_mask, token_start_mask

    @classmethod
    def from_corpus(cls, corpus, min_freq=1):
        words = Counter(word for seq in corpus.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        chars = list({char for seq in corpus.words for char in ''.join(seq)})
        rels = list({rel for seq in corpus.rels for rel in seq})
        vocab = cls(words, chars, rels)

        return vocab
