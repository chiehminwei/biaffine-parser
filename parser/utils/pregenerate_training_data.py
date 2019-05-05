from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve

from random import random, randrange, randint, shuffle, choice, sample
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
import json

from collections import namedtuple

Sentence = namedtuple(typename='Sentence',
                      field_names=['ID', 'FORM', 'LEMMA', 'CPOS',
                                   'POS', 'FEATS', 'HEAD', 'DEPREL',
                                   'PHEAD', 'PDEPREL'])

class DocumentDatabase:
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        # Uses the current iteration counter to ensure we don't sample the same doc twice
        if sentence_weighted:
            # With sentence weighting, we sample docs proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        assert sampled_doc_index != current_idx
        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indices.append(i)

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)
    mask_indices = sorted(sample(cand_indices, num_to_mask))
    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab_list)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels


def create_instances_from_document(
        doc_database, doc_idx, max_seq_length,
        masked_lm_prob, max_predictions_per_seq, vocab_list, tokenizer, rel_dict):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    document = doc_database[doc_idx]
    
    instances = []
    exceeding_count = 0
    for sentence in document:
        words = sentence.FORM
        heads = sentence.HEAD
        rels = sentence.DEPREL

        tokens = []
        word_start_mask = []
        word_end_mask = []
        sent_arcs = []
        sent_rels = []

        tokens.append("[CLS]")
        word_start_mask.append(1)
        word_end_mask.append(1)
        sent_arcs.append(0)
        sent_rels.append(0)

        for word, head, rel in zip(words, heads, rels):
            word_tokens = tokenizer.tokenize(word)
            for _ in range(len(word_tokens)):
                word_start_mask.append(0)
                word_end_mask.append(0)
                sent_arcs.append(head)
                sent_rels.append(rel_dict.get(rel, 0))
            word_start_mask[len(tokens)] = 1
            word_end_mask[-1] = 1
            tokens.extend(word_tokens)

        tokens.append("[SEP]")
        sent_arcs.append(0)
        sent_rels.append(0)
        word_start_mask.append(1)
        word_end_mask.append(1)

        if len(tokens) > max_seq_length:
            print('Skipping sentence longer than max_seq_length {}.'.format(max_seq_length))
            exceeding_count += 1
            continue

        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_list)

        instance = {
            "tokens": tokens,
            "word_start_mask": word_start_mask,
            "word_end_mask": word_end_mask,
            "arc_ids": sent_arcs,
            "rel_ids": sent_rels,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels}

        instances.append(instance)

    print('Total number of sentences: {}'.format(len(document)))
    print('Number of sentences exceeding max seq length of {}: {}'.format(max_seq_length, exceeding_count))

    return instances


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual", "bert-base-chinese"])
    parser.add_argument("--do_lower_case", action="store_true")
    
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")

    parser.add_argument("--epochs_to_generate", type=int, default=60,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=128)

    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    vocab_list = list(tokenizer.vocab.keys())
    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        rels = set()
        with args.train_corpus.open() as f:
            doc = []            
            lines = f.readlines()
            for i, line in tqdm(enumerate(lines), desc="Loading Dataset"):
                if line[0] == '#':
                    start += 1
                if len(line) <= 1:
                    sentence = Sentence(*zip(*[l.split('\t') for l in lines[start:i] if "." not in l.split('\t')[0] and "-" not in l.split('\t')[0]]))
                    rels.add(sentence.DEPREL)
                    doc.append(sentence)
                    start = i + 1
            docs.add_document(doc)

        rel_dict = {rel: i for i, rel in enumerate(sorted(rels))}

        args.output_dir.mkdir(exist_ok=True)
        for epoch in trange(args.epochs_to_generate, desc="Epoch"):
            epoch_filename = args.output_dir / f"epoch_{epoch}.json"
            num_instances = 0
            with epoch_filename.open('w') as epoch_file:
                for doc_idx in trange(len(docs), desc="Document"):
                    doc_instances = create_instances_from_document(
                        docs, doc_idx, max_seq_length=args.max_seq_len,
                        masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                        vocab_list=vocab_list, tokenizer=tokenizer, rel_dict=rel_dict)
                    doc_instances = [json.dumps(instance) for instance in doc_instances]
                    for instance in doc_instances:
                        epoch_file.write(instance + '\n')
                        num_instances += 1
            metrics_file = args.output_dir / f"epoch_{epoch}_metrics.json"
            with metrics_file.open('w') as metrics_file:
                metrics = {
                    "num_training_examples": num_instances,
                    "max_seq_len": args.max_seq_len
                }
                metrics_file.write(json.dumps(metrics))


if __name__ == '__main__':
    main()