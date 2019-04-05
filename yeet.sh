#! /bin/bash


path="all";
for conllx_file in ${path}/*.conllx; do
	filename=$(echo $conllx_file | cut -d '/' -f 2 | cut -d '.' -f 1)
	echo ' '
	echo '***************************************'
	echo $filename
	echo '***************************************'
	echo ' '
	bash CUDA_VISIBLE_DEVICES=2,3 python run.py train --ftrain=$conllx_file --ftrain_cache=all_trainset/$filename --vocab=all_vocab/$filename
	echo '***************************************'
	echo ' '
done
