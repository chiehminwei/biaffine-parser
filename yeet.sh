#! /bin/bash


path="all";
for conllx_file in ${path}/*.conllx; do
	filename=$(echo $conllx_file | cut -d '/' -f 2 | cut -d '.' -f 1)
	echo $filename
	# bash CUDA_VISIBLE_DEVICES=2,3 python run.py train -c=gs:/dsfsdfs --ftrain=$conllx_file --ftrain_cache= --fdev_cache=
done
