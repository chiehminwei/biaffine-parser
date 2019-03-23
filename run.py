# -*- coding: utf-8 -*-

import os
import argparse
from parser.commands import Evaluate, Predict, Train

import torch
import logging

if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    # torch.set_printoptions(threshold=10000)
    
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )
    subparsers = parser.add_subparsers(title='Commands')
    subcommands = {
        'evaluate': Evaluate(),
        'predict': Predict(),
        'train': Train()
    }
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        # subparser.add_argument('--device', '-d', default='-1',
        #                        help='ID of GPU to use')
        subparser.add_argument('--seed', '-s', default=1, type=int,
                               help='seed for generating random numbers')
        subparser.add_argument('--threads', '-t', default=4, type=int,
                               help='max num of threads')
        subparser.add_argument('--file', '-f', default='model.pt',
                               help='path to model file')
        subparser.add_argument('--vocab', '-v', default='vocab.pt',
                               help='path to vocabulary file')
        subparser.add_argument('--cloud_address', '-c',
                               default='gs://bert-chinese-mine/biaffine/',
                               help='path to Google Cloud Storage')
    args = parser.parse_args()

    print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    # print(f"Set the device with ID {args.device} visible")
    
    # torch.set_num_threads(args.num_threads)
    torch.set_num_threads(6)
    torch.manual_seed(args.seed)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    args.func(args)
