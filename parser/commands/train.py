# -*- coding: utf-8 -*-

from parser import BiaffineParser, Model
from parser.utils import Corpus, TextDataset, Vocab, collate_fn

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import Config

import os
import subprocess
from datetime import datetime, timedelta

from pathlib import Path
import logging
import json


class Train(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--pregenerated_data', type=Path)
        subparser.add_argument("--bert_model", type=str, required=True,
                choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                         "bert-base-multilingual-cased", "bert-base-multilingual-uncased", "bert-base-chinese"])
        subparser.add_argument("--do_lower_case", action="store_true")
        subparser.add_argument("--reduce_memory", action="store_true",
                            help="Store training data as on-disc memmaps to massively reduce memory usage")
        subparser.add_argument("--resume_training", action="store_true")
        subparser.add_argument("--warmup_proportion",
                            default=0.1,
                            type=float,
                            help="Proportion of training to perform linear learning rate warmup for. "
                                 "E.g., 0.1 = 10%% of training.")
        subparser.add_argument('--ftrain', default='data/train.conllx',
                               help='path to raw train file')
        subparser.add_argument('--fdev', default='data/dev.conllx',
                               help='path to raw dev file')
        subparser.add_argument('--ftest', default='data/test.conllx',
                               help='path to raw test file')
        subparser.add_argument('--ftrain_cache', default='data/binary/trainset',
                               help='path to train file cache')
        subparser.add_argument('--fdev_cache', default='data/binary/devset',
                               help='path to dev file cache')
        subparser.add_argument('--ftest_cache', default='data/binary/testset',
                               help='path to test file cache')
        subparser.add_argument('--train_lm', action="store_true")
        subparser.add_argument('--step_decay_factor', type=float, default=0.5)
        subparser.add_argument('--step_decay_patience', type=int, default=5)
        subparser.set_defaults(func=self)

        return subparser

    def __call__(self, args):

        if args.checkpoint_dir.is_dir() and list(args.checkpoint_dir.iterdir()):
            logging.warning(f"Output directory ({args.checkpoint_dir}) already exists and is not empty!")
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        try:
            index = args.vocab.rfind('/')
            if index > -1:
                save_dir = args.vocab[:index]
                os.makedirs(save_dir, parents=True)
        except FileExistsError:
            # directory already exists
            pass


        num_data_epochs = 0
        num_train_optimization_steps = 0
        if args.train_lm:
            assert args.pregenerated_data.is_dir(), \
            "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"

            samples_per_epoch = []
            for i in range(Config.epochs):
                epoch_file = args.pregenerated_data / f"epoch_{i}.json"
                metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
                if epoch_file.is_file() and metrics_file.is_file():
                    metrics = json.loads(metrics_file.read_text())
                    samples_per_epoch.append(metrics['num_training_examples'])
                else:
                    if i == 0:
                        exit("No training data was found!")
                    print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({Config.epochs}).")
                    print("This script will loop over the available data, but training diversity may be negatively impacted.")
                    num_data_epochs = i
                    break
            else:
                num_data_epochs = Config.epochs

        
            total_train_examples = 0
            for i in range(Config.epochs):
                # The modulo takes into account the fact that we may loop over limited epochs of data
                total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

            num_train_optimization_steps = int(
                total_train_examples / Config.batch_size / Config.gradient_accumulation_steps)
            if args.distributed:
                num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        train = Corpus.load(args.ftrain)
        dev = Corpus.load(args.fdev)
        test = Corpus.load(args.ftest)       
        if not os.path.isfile(args.vocab):
            vocab = Vocab.from_corpus(corpus=train, min_freq=2, bert_model=args.bert_model, do_lower_case=args.do_lower_case)
            torch.save(vocab, args.vocab) 
        else:
            vocab = torch.load(args.vocab)

        if args.local_rank == 0:
            logging.info(vocab)

        # Load train, dev, test
        if not os.path.isfile(args.ftrain_cache):
            if args.local_rank == 0:
                logging.info('Loading trainset from scratch.')
            trainset = TextDataset(vocab.numericalize(train, args.ftrain_cache))
        else:
            if args.local_rank == 0:
                logging.info('Loading trainset from checkpoint.')
            trainset = TextDataset(torch.load(args.ftrain_cache))
        if args.local_rank == 0:
            logging.info('***Trainset loaded at {}***'.format(datetime.now()))

        if not os.path.isfile(args.fdev_cache):
            if args.local_rank == 0:
                logging.info('Loading devset from scratch.')
            devset = TextDataset(vocab.numericalize(dev, args.fdev_cache))
        else:
            if args.local_rank == 0:
                logging.info('Loading devset from checkpoint.')
            devset = TextDataset(torch.load(args.fdev_cache))
        if args.local_rank == 0:
            logging.info('***Devset loaded at {}***'.format(datetime.now()))

        if not os.path.isfile(args.ftest_cache):
            if args.local_rank == 0:
                logging.info('Loading testset from scratch.')
            testset = TextDataset(vocab.numericalize(test, args.ftest_cache))
        else:
            if args.local_rank == 0:
                logging.info('Loading testset from checkpoint.')
            testset = TextDataset(torch.load(args.ftest_cache))
        if args.local_rank == 0:
            logging.info('***Testset loaded at {}***'.format(datetime.now()))

        # set up data loaders
        train_sampler = None        
        if args.distributed:
            train_sampler = DistributedSampler(trainset)
        train_loader = DataLoader(dataset=trainset,
                                batch_size=Config.batch_size // Config.gradient_accumulation_steps,
                                num_workers=0,
                                shuffle=True,
                                sampler=train_sampler,
                                collate_fn=collate_fn)

        dev_sampler = None        
        if args.distributed:
            dev_sampler = DistributedSampler(devset)
        dev_loader = DataLoader(dataset=devset,
                                batch_size=Config.batch_size,
                                num_workers=0,
                                shuffle=False,
                                sampler=dev_sampler,
                                collate_fn=collate_fn)

        test_sampler = None        
        if args.distributed:
            test_sampler = DistributedSampler(testset)
        test_loader = DataLoader(dataset=testset,
                                batch_size=Config.batch_size,
                                num_workers=0,
                                shuffle=False,
                                sampler=test_sampler,
                                collate_fn=collate_fn)

        if args.local_rank == 0:
            if args.train_lm:
                logging.info(f"  Num train examples = {total_train_examples}")
                logging.info("  Batch size = %d", Config.batch_size)
                logging.info("  Num steps = %d", num_train_optimization_steps)
            else:
                logging.info(f"{'train:':6} {len(trainset):5} sentences in total.")
                logging.info(f"{'dev:':6} {len(devset):5} sentences in total.")
                logging.info(f"{'test:':6} {len(testset):5} sentences in total.")
                logging.info("  Batch size = %d", Config.batch_size)

        params = {
            'n_bert_hidden': Config.n_bert_hidden,
            'bert_dropout': Config.bert_dropout,
            'n_lstm_layers': Config.n_lstm_layers,
            'n_lstm_hidden': Config.n_lstm_hidden,
            'lstm_dropout': Config.lstm_dropout,
            'n_mlp_arc': Config.n_mlp_arc,
            'n_mlp_rel': Config.n_mlp_rel,
            'mlp_dropout': Config.mlp_dropout,
            'n_rels': vocab.n_rels,
            'bert_model': args.bert_model,
            'use_lstm': args.use_lstm,
            'use_pos': args.use_pos,
            'n_tags': vocab.n_tags,
            'n_tag_embed': Config.n_tag_embed,
        }
        if args.local_rank == 0:
            for k, v in params.items():
                if args.local_rank == 0:
                    logging.info(f"  {k}: {v}")
        network = BiaffineParser(params)
        if torch.cuda.is_available() and not args.no_cuda:
          network.to(torch.device('cuda'))

        # if args.local_rank == 0:
        #     logging.info(f"{network}\n")

        last_epoch = 0
        max_metric = 0.0

        # Start training from checkpoint if one exists
        state = None
        if args.resume_training and os.path.isfile(args.file):
            if args.local_rank == 0:
                logging.info('Resume training from checkpoint.')
            state = torch.load(args.file, map_location='cpu')
            last_epoch = state['last_epoch']
            network = network.load(args.file, args.cloud_address, args.local_rank)

        n_gpu = torch.cuda.device_count()
        if args.distributed:
            n_gpu = 1
            try:
                from apex.parallel import DistributedDataParallel
            except ImportError:
                logging.exception("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            if args.local_rank == 0:
                logging.info('Distirbuted training enabled.')
            network = DistributedDataParallel(network)

        elif n_gpu > 1 and not args.no_cuda:
            if args.local_rank == 0:
                logging.info('Using {} GPUs for data parallel training'.format(torch.cuda.device_count()))
            network = torch.nn.DataParallel(network)

        model = Model(vocab, network)
        if args.resume_training and os.path.isfile(args.file):
            try:
                if args.local_rank == 0:
                    logging.info('Resume training for optimizer')
                max_metric = state['max_metric']
                model.optimizer.load_state_dict(state['optimizer'])
            except:
                logging.warning('Optimizer failed to load')


        model(loaders=(train_loader, dev_loader, test_loader),
              epochs=Config.epochs,
              num_data_epochs=num_data_epochs,
              patience=Config.patience,
              lr=Config.lr,
              t_total=num_train_optimization_steps,
              last_epoch=last_epoch,
              cloud_address=args.cloud_address,
              args=args,
              batch_size=Config.batch_size,
              gradient_accumulation_steps=Config.gradient_accumulation_steps,
              max_metric=max_metric)
