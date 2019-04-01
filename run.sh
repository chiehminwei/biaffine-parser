#!/bin/bash
python create_datasets.py -c=gs://bert-chinese-mine/wut_wut_wut/
python -m torch.distributed.launch --nproc_per_node=4 run.py train -c=gs://bert-chinese-mine/wut_wut_wut/