#!/bin/bash
python create_datasets.py --bert_model bert-base-cased && CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run.py train --checkpoint_dir checkpoints --pregenerated_data training --bert_model bert-base-cased --save_log_to_file --reduce_memory --train_lm
