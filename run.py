# -*- coding: utf-8 -*-

import os
import argparse
from parser.commands import Evaluate, Predict, Train

from pathlib import Path
import torch
import logging
import random
import numpy as np


if __name__ == '__main__':
    # torch.set_printoptions(threshold=10000)
    
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )
    parser.add_argument('--local_rank', '-l', default=0, type=int,
                         help='local rank for distributed training')

    subparsers = parser.add_subparsers(title='Commands')
    subcommands = {
        'evaluate': Evaluate(),
        'predict': Predict(),
        'train': Train()
    }
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        
        subparser.add_argument('--seed', '-s', default=1, type=int,
                               help='seed for generating random numbers')
        subparser.add_argument('--checkpoint_dir', type=Path, required=True)
        subparser.add_argument('--vocab', '-v', default='vocab.pt',
                               help='path to vocabulary file')
        subparser.add_argument('--cloud_address', '-c',
                               default="you dont want to use this bro trust me too annoying",
                               help='path to Google Cloud Storage')
        subparser.add_argument("--no_cuda",
                            action='store_true',
                            help="Whether to use CUDA when available")
        subparser.add_argument("--save_log_to_file",
                            action='store_true',
                            help="Whether to log to file")
        subparser.add_argument('--logdir', default='logs', type=Path,
                                   help='Directory to save log')
        subparser.add_argument('--use_lstm',
                            action='store_true',
                            help="Whether to use BiLSTM and freeze BERT embeddings")
        
    args = parser.parse_args()
    log_format = '%(asctime)-10s: %(message)s'
    if args.save_log_to_file:
        if args.local_rank == 0:
            args.logdir.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(filename="logs/log", filemode='w', format=log_format, level=logging.INFO)
    else:
        if args.local_rank == 0:
            logging.basicConfig(format=log_format, level=logging.INFO)
    
    # FOR DISTRIBUTED:  If we are running under torch.distributed.launch,
    # the 'WORLD_SIZE' environment variable will also be set automatically.
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    n_gpu = torch.cuda.device_count()
    if args.distributed:
        # FOR DISTRIBUTED:  Set the device according to local_rank.
        torch.cuda.set_device(args.local_rank)

        # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
        # environment variables, and requires that you use init_method=`env://`.
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        n_gpu = 1

    torch.backends.cudnn.benchmark = True

    if args.local_rank == 0:
        logging.info(f"Set the seed for generating random numbers to {args.local_rank}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.local_rank)
    
    if args.local_rank == 0:
        logging.info("CUDNN VERSION: {}".format(torch.backends.cudnn.version()))
    
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    args.func(args)
