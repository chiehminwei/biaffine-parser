# -*- coding: utf-8 -*-

from parser import BiaffineParser, Model
from parser.utils import Corpus, TextDataset, Vocab, collate_fn

import torch
from torch.utils.data import DataLoader

from config import Config

import os
import subprocess
from datetime import datetime, timedelta


class Train(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--ftrain', default='data/train.conllx',
                               help='path to train file')
        subparser.add_argument('--fdev', default='data/dev.conllx',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='data/test.conllx',
                               help='path to test file')
        subparser.set_defaults(func=self)

        return subparser

    def __call__(self, args):
        print("Preprocess the data. {}".format(datetime.now()))
        
        train = Corpus.load(args.ftrain)
        dev = Corpus.load(args.fdev)
        test = Corpus.load(args.ftest)

        if not os.path.isfile(args.vocab):
            FNULL = open(os.devnull, 'w')
            cloud_address = os.path.join(args.cloud_address, args.vocab)
            subprocess.call(['gsutil', 'cp', cloud_address, args.vocab],
                            stdout=FNULL, stderr=subprocess.STDOUT)
        if not os.path.isfile(args.vocab):
            vocab = Vocab.from_corpus(corpus=train, min_freq=2)
            torch.save(vocab, args.vocab)
            FNULL = open(os.devnull, 'w')
            cloud_address = os.path.join(args.cloud_address, args.vocab)
            subprocess.call(['gsutil', 'cp', args.vocab, cloud_address],
                            stdout=FNULL, stderr=subprocess.STDOUT)
        else:
            vocab = torch.load(args.vocab)

        print(vocab)

        print("Load the dataset. {}".format(datetime.now()))

        if not os.path.isfile('trainset'):
            trainset = TextDataset(vocab.numericalize(train, 'trainset'))
        else:
            trainset = TextDataset(torch.load('trainset'))
        if not os.path.isfile('devset'):
            devset = TextDataset(vocab.numericalize(dev, 'devset'))
        else:
            devset = TextDataset(torch.load('devset'))
        if not os.path.isfile('testset'):
            testset = TextDataset(vocab.numericalize(test, 'testset'))
        else:
            testset = TextDataset(torch.load('testset'))
        
        # set the data loaders
        train_loader = DataLoader(dataset=trainset,
                                  batch_size=Config.batch_size // Config.gradient_accumulation_steps,
                                  shuffle=True,
                                  collate_fn=collate_fn)
        dev_loader = DataLoader(dataset=devset,
                                batch_size=Config.batch_size,
                                collate_fn=collate_fn)
        test_loader = DataLoader(dataset=testset,
                                 batch_size=Config.batch_size,
                                 collate_fn=collate_fn)
        print(f"  size of trainset: {len(trainset)}")
        print(f"  size of devset: {len(devset)}")
        print(f"  size of testset: {len(testset)}")

        print("Create the model. {}".format(datetime.now()))
        params = {
            'n_words': vocab.n_train_words,
            'n_chars': vocab.n_chars,
            'word_dropout': Config.word_dropout,
            'n_bert_hidden': Config.n_bert_hidden,
            'bert_dropout': Config.bert_dropout,
            'n_mlp_arc': Config.n_mlp_arc,
            'n_mlp_rel': Config.n_mlp_rel,
            'mlp_dropout': Config.mlp_dropout,
            'n_rels': vocab.n_rels,
            'pad_index': vocab.pad_index
        }
        for k, v in params.items():
            print(f"  {k}: {v}")
        network = BiaffineParser(params)
        if torch.cuda.is_available():
            network = network.cuda()
        print(f"{network}\n")

        last_epoch = 0
        # Start training from checkpoint if one exists
        if not os.path.isfile(args.file):
            FNULL = open(os.devnull, 'w')
            cloud_address = os.path.join(args.cloud_address, args.file)
            subprocess.call(['gsutil', 'cp', cloud_address, args.file], stdout=FNULL, stderr=subprocess.STDOUT)
        if os.path.isfile(args.file):
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            state = torch.load(args.file, map_location=device)
            last_epoch = state['last_epoch']
            network = network.load(args.file, args.cloud_address)

        if torch.cuda.device_count() > 1:
            print('Using {} GPUs to train'.format(torch.cuda.device_count()))
            network = torch.nn.DataParallel(network)

        model = Model(vocab, network)
        model(loaders=(train_loader, dev_loader, test_loader),
              epochs=Config.epochs,
              patience=Config.patience,
              lr=Config.lr,
              betas=(Config.beta_1, Config.beta_2),
              epsilon=Config.epsilon,
              weight_decay=Config.weight_decay,
              annealing=lambda x: Config.decay ** (x / Config.decay_steps),
              file=args.file,
              last_epoch=last_epoch,
              cloud_address=args.cloud_address,
              gradient_accumulation_steps=Config.gradient_accumulation_steps)
