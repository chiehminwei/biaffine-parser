# -*- coding: utf-8 -*-

from collections import Counter, defaultdict

import regex
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from tqdm import tqdm
import numpy as np
import unicodedata
import os
import logging


class Vocab(object):
    PAD = '<PAD>'

    def __init__(self, words, chars, rels, tags, bert_model, do_lower_case):
        self.pad_index = 0

        self.words = [self.PAD] + sorted(words)
        self.chars = [self.PAD] + sorted(chars)
        self.rels = sorted(rels)
        self.tags = sorted(tags)

        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.char_dict = {char: i for i, char in enumerate(self.chars)}
        self.rel_dict = {rel: i for i, rel in enumerate(self.rels)}
        self.tag_dict = {tag: i for i, tag in enumerate(self.tags)}

        # ids of punctuation that appear in words
        self.puncts = set(sorted(i for word, i in self.word_dict.items()
                                if regex.match(r'\p{P}+$', word)))

        self.n_words = len(self.words)
        self.n_chars = len(self.chars)
        self.n_rels = len(self.rels)
        self.n_tags = len(self.tags)
        self.n_train_words = self.n_words

        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
        self.bert = BertModel.from_pretrained(bert_model).to('cuda')
        self.bert.eval()

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
        tags_numerical = []
        token_start_mask = []
        attention_mask = []
        offending_set = set()
        symbol_set = set()
        empty_words = set()
        len_dict = defaultdict(int)
        sent_count = 0
        exceeding_count = 0
        kkk = 0
        with tqdm(total=len(corpus.words)) as pbar:
            for words, arcs, rels, tags in tqdm(zip(corpus.words, corpus.heads, corpus.rels, corpus.tags)):
                pbar.update(1)
                kkk += 1
                sentence_token_ids = []
                sentence_arc_ids = []
                sentence_rel_ids = []
                sentennce_tag_ids = []
                token_starts = []
                attentions = []


                
                for word, arc, rel, tag in zip(words, arcs, rels, tags):
                    # skip <ROOT>
                    if word == '<ROOT>':
                        word = '[CLS]'

                    tokens = self.tokenizer.tokenize(word)                
                    if len(tokens) > 0:
                        ids = self.tokenizer.convert_tokens_to_ids(tokens)

                        # take care of punctuation
                        if regex.match(r'\p{P}+$', word):
                            for token_id in ids:
                                self.puncts.add(token_id)

                        # log any unknown words
                        if '[UNK]' in tokens:
                            for offending_char in word:
                                token = self.tokenizer.tokenize(offending_char)
                                if '[UNK]' in token:
                                    if unicodedata.category(offending_char) != 'So':
                                        offending_set.add(offending_char)
                                    else:
                                        symbol_set.add(offending_char)
                            
                        # main thing to do
                        sentence_token_ids.extend(ids)
                        sentence_arc_ids.extend([arc])
                        sentence_rel_ids.extend([self.rel_dict.get(rel, 0)])
                        sentennce_tag_ids.extend([self.tag_dict.get(tag, 0)])
                        token_starts.extend([1] + [0] * (len(tokens) - 1))
                        attentions.extend([1])

                    # take care of empty tokens
                    else:
                        empty_words.add(word)
                        continue

                sent_count += 1                
                # Skip too long sentences
                if len(sentence_token_ids) > 128:
                    exceeding_count += 1
                    continue
                
                sentence_token_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]']) + sentence_token_ids + self.tokenizer.convert_tokens_to_ids(['[SEP]'])
                token_starts = [0] + token_starts + [0]

                tokens_tensor = torch.tensor([sentence_token_ids]).to('cuda')
                segments_tensors = torch.tensor([0 for i in sentence_token_ids]).to('cuda')

                # BERT9-12
                layers = []
                with torch.no_grad():
                    bert_output, _ = self.bert(tokens_tensor, segments_tensors)
                del _
                for layer in range(8, 12):
                  layer_masked = []
                  for i, mask in enumerate(token_starts):
                    if mask == 1:
                      layer_masked.append(bert_output[layer][0][i])
                  layer_masked = torch.stack(layer_masked)
                  layers.append(layer_masked)
                bert_embeddings = torch.sum(torch.stack(layers), dim=0)

                token_starts = token_starts[1:-1]
                
                words_numerical.append(bert_embeddings)
                arcs_numerical.append(torch.tensor(sentence_arc_ids))
                rels_numerical.append(torch.tensor(sentence_rel_ids))
                tags_numerical.append(torch.tensor(sentennce_tag_ids))

                token_start_mask.append(torch.ByteTensor(attentions))
                attention_mask.append(torch.ByteTensor(attentions))
                
                if kkk < 3:
                    print(token_start_mask)
                    print(words)
                    print(sum(token_starts))
                    print(layers[0].shape)
                    print(words_numerical[-1].shape)
                    print(arcs_numerical[-1].shape)
                    print(rels_numerical[-1].shape)
                    print(tags_numerical[-1].shape)

        if offending_set: 
            logging.warning('WARNING: The following non-symbol characters are unknown to BERT:')
            try:
                logging.warning(offending_set)
            except:
                pass
        if symbol_set:
            logging.warning('WARNING: The following symbol characters are unknown to BERT:')
            try:         
                logging.warning(symbol_set)
            except:
                pass
        if empty_words:
            logging.warning('WARNING: The following characters are empty after going through tokenizer:')
            try:
                logging.warning(empty_words)
            except:
                pass
        if save_name:
            try:
                index = save_name.rfind('/')
                if index > -1:
                    save_dir = save_name[:index]
                    os.makedirs(save_dir)
            except FileExistsError:
                pass
            torch.save((words_numerical, attention_mask, token_start_mask, arcs_numerical, rels_numerical, tags_numerical), save_name)
        
        logging.info('Total number of sentences: {}'.format(sent_count))
        logging.info('Number of sentences exceeding max seq length of 128: {}'.format(exceeding_count))

        return words_numerical, attention_mask, token_start_mask, arcs_numerical, rels_numerical, tags_numerical


    def numericalize_first_token(self, corpus, save_name=None):
        words_numerical = []
        arcs_numerical = []
        rels_numerical = []
        tags_numerical = []
        token_start_mask = []
        attention_mask = []
        offending_set = set()
        symbol_set = set()
        empty_words = set()
        len_dict = defaultdict(int)
        sent_count = 0
        exceeding_count = 0
        kkk = 0
        for words, arcs, rels, tags in zip(corpus.words, corpus.heads, corpus.rels, corpus.tags):
            kkk += 1
            sentence_token_ids = []
            sentence_arc_ids = []
            sentence_rel_ids = []
            sentennce_tag_ids = []
            token_starts = []
            attentions = []
            words = ['[CLS]'] + words + ['[SEP]']
            arcs = [0] + arcs + [0]
            rels = ['<ROOT>'] + rels + ['<ROOT>']
            tags = ['<ROOT>'] + tags + ['<ROOT>']
            for word, arc, rel, tag in zip(words, arcs, rels, tags):
                # skip <ROOT>
                if word == '<ROOT>':
                    continue
                
                # take care of some idiosyncracies
                # if word == '`':
                #     word = "'"
                # if word == '``':
                #     word = '"'
                # if word == "''":
                #     word = '"'
                # if word == "non-``":
                #     word = 'non-"'
                # word = word.replace('“', '"')
                # word = word.replace('”', '"')
                # word = word.replace("`", "'")
                # word = word.replace("’", "'")
                # word = word.replace("‘", "'")
                # word = word.replace("'", "'")
                # word = word.replace("´", "'")
                # word = word.replace("…", "...")
                # word = word.replace("–", "-")
                # word = word.replace('—', '-')


                tokens = self.tokenizer.tokenize(word)                
                if len(tokens) > 0:
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
                        
                    # main thing to do
                    sentence_token_ids.extend(ids)
                    sentence_arc_ids.extend([arc] * len(tokens))
                    sentence_rel_ids.extend([self.rel_dict.get(rel, 0)] * len(tokens))
                    sentennce_tag_ids.extend([self.tag_dict.get(tag, 0)] * len(tokens))
                    token_starts.extend([1] + [0] * (len(tokens) - 1))
                    attentions.extend([1] * len(tokens))

                # take care of empty tokens
                else:
                    # print('\noffending word: ', word)
                    # print('empty words: ', ' '.join(words))
                    empty_words.add(word)
                    continue
                
            # error checking for lengths
            # if kkk < 3:
            #     print(words)
            #     print(self.tokenizer.convert_ids_to_tokens(sentence_token_ids))
            len_sentence_token_ids = len(sentence_token_ids)
            if len_sentence_token_ids == 1: continue
            len_sentence_arc_ids = len(sentence_arc_ids)
            len_sentence_rel_ids = len(sentence_rel_ids)
            len_token_starts = len(token_starts)
            len_attentions = len(attentions)
            len_sentence_tag_ids = len(sentennce_tag_ids)
            if not (len_sentence_token_ids == len_sentence_arc_ids == len_sentence_rel_ids == len_token_starts == len_attentions == len_sentence_tag_ids):
                logging.debug(words)
                logging.debug(arcs)
                logging.debug(rels)
                logging.debug('len_sentence_token_ids: ', len_sentence_token_ids)
                logging.debug('len_sentence_arc_ids', len_sentence_arc_ids)
                logging.debug('len_sentence_rel_ids', len_sentence_rel_ids)
                logging.debug('len_token_starts', len_token_starts)
                logging.debug('len_attentions', len_attentions)
                logging.debug('len_sentence_tag_ids', len_sentence_tag_ids)
                raise RuntimeError("Lengths don't match up.")

            # Skip too long sentences
            len_dict[len_sentence_token_ids] += 1
            if len_sentence_token_ids > 128:
                exceeding_count += 1
                continue
            sent_count += 1                

            words_numerical.append(torch.tensor(sentence_token_ids))
            arcs_numerical.append(torch.tensor(sentence_arc_ids))
            rels_numerical.append(torch.tensor(sentence_rel_ids))
            tags_numerical.append(torch.tensor(sentennce_tag_ids))
            token_start_mask.append(torch.ByteTensor(token_starts))
            attention_mask.append(torch.ByteTensor(attentions))

        if offending_set: 
            logging.warning('WARNING: The following non-symbol characters are unknown to BERT:')
            try:
                logging.warning(offending_set)
            except:
                pass
        if symbol_set:
            logging.warning('WARNING: The following symbol characters are unknown to BERT:')
            try:         
                logging.warning(symbol_set)
            except:
                pass
        if empty_words:
            logging.warning('WARNING: The following characters are empty after going through tokenizer:')
            try:
                logging.warning(empty_words)
            except:
                pass
        if save_name:
            try:
                index = save_name.rfind('/')
                if index > -1:
                    save_dir = save_name[:index]
                    os.makedirs(save_dir)
            except FileExistsError:
                # directory already exists
                pass
            torch.save((words_numerical, attention_mask, token_start_mask, arcs_numerical, rels_numerical, tags_numerical), save_name)
        
        logging.info('Total number of sentences: {}'.format(sent_count))
        logging.info('Number of sentences exceeding max seq length of 128: {}'.format(exceeding_count))

        return words_numerical, attention_mask, token_start_mask, arcs_numerical, rels_numerical, tags_numerical
        
    def numericalize_sentences(self, sentences):
        words_numerical = []
        token_start_mask = []
        attention_mask = []
        offending_set = set()
        symbol_set = set()
        empty_words = set()
        exceeding_count = 0
        for sentence in sentences:
            sentence_token_ids = []
            token_starts = []
            attentions = []
            sentence = ['[CLS]'] + sentence + ['[SEP]']
            for word in sentence:
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
                    
                    # Keep track of punctuation
                    if regex.match(r'\p{P}+$', word):
                        for token_id in ids:
                            self.puncts.add(token_id)

                    # log any unknown words
                    if '[UNK]' in tokens:
                        for offending_char in word:
                            token = self.tokenizer.tokenize(offending_char)
                            if unicodedata.category(offending_char) != 'So':
                                offending_set.add(offending_char)
                            else:
                                symbol_set.add(offending_char)
                        
                    sentence_token_ids.extend(ids)
                    token_starts.extend([1] + [0] * (len(tokens) - 1))
                    attentions.extend([1] * len(tokens))
                
                # take care of empty tokens
                else:
                    empty_words.add(word)
                    continue

            # Skip too long sentences
            len_sentence_token_ids = len(sentence_token_ids)
            if len_sentence_token_ids > 128:
                exceeding_count += 1
                continue

            words_numerical.append(torch.tensor(sentence_token_ids))
            attention_mask.append(torch.ByteTensor(attentions))
            token_start_mask.append(torch.ByteTensor(token_starts))    
        
        return words_numerical, attention_mask, token_start_mask

    def numericalize_tags(self, corpus):
        words_numerical = []
        words_total = []
        tags_total = []
        token_start_mask = []
        attention_mask = []
        offending_set = set()
        symbol_set = set()
        empty_words = set()
        exceeding_count = 0
        for sentence, words, tags in zip(corpus.words, corpus.words, corpus.tags):
            # skip <ROOT>
            words = words[1:]
            tags = tags[1:]

            sentence_token_ids = []
            token_starts = []
            attentions = []
            sentence = ['[CLS]'] + sentence + ['[SEP]']
            for word in sentence:
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
                    
                    # Keep track of punctuation
                    if regex.match(r'\p{P}+$', word):
                        for token_id in ids:
                            self.puncts.add(token_id)

                    # log any unknown words
                    if '[UNK]' in tokens:
                        for offending_char in word:
                            token = self.tokenizer.tokenize(offending_char)
                            if unicodedata.category(offending_char) != 'So':
                                offending_set.add(offending_char)
                            else:
                                symbol_set.add(offending_char)
                        
                    sentence_token_ids.extend(ids)
                    token_starts.extend([1] + [0] * (len(tokens) - 1))
                    attentions.extend([1] * len(tokens))
                
                # take care of empty tokens
                else:
                    empty_words.add(word)
                    continue

            # Skip too long sentences
            len_sentence_token_ids = len(sentence_token_ids)
            if len_sentence_token_ids > 128:
                exceeding_count += 1
                continue

            words_numerical.append(torch.tensor(sentence_token_ids))
            attention_mask.append(torch.ByteTensor(attentions))
            token_start_mask.append(torch.ByteTensor(token_starts))
            words_total.append(words)
            tags_total.append(tags)
            
        return words_numerical, attention_mask, token_start_mask, words_total, tags_total

    @classmethod
    def from_corpus(cls, corpus, bert_model, do_lower_case, min_freq=1):
        words = Counter(word for seq in corpus.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        chars = list({char for seq in corpus.words for char in ''.join(seq)})
        rels = list({rel for seq in corpus.rels for rel in seq})
        tags = list({tag for seq in corpus.tags for tag in seq})
        vocab = cls(words, chars, rels, tags, bert_model, do_lower_case)

        return vocab
