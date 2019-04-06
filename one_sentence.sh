#! /bin/bash


path="debugging_one_sentence";
for conllx_file in ${path}/*.conllx; do
        filename=$(echo $conllx_file | cut -d '/' -f 2 | cut -d '.' -f 1)
        echo ' '
        echo '***************************************'
        echo $filename
        echo '***************************************'
        echo ' '
        CUDA_VISIBLE_DEVICES=2,3 python run.py train --ftrain=$conllx_file --ftrain_cache=$path/${filename}.cache --vocab=$path/${filename}.vocab && echo $filename >> success_debug || echo $filename >> failure_debug
        echo '***************************************'
        echo ' '
done