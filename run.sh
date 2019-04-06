#!/bin/bash
python create_datasets.py -c=gs://udontwantcloud && CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 run.py train