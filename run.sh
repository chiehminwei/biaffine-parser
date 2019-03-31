#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 run.py train