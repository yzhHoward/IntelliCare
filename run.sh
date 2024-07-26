#!/bin/bash
dataset="mimiciii"
task=("mortality" "readmission" "length_of_stay")
text_path="export/mimiciii/llama-cluster-8x"
text_num=8
suffix="_test"
world_size=4
backbone=("concare")
seed=(42 3407)

gpu=0
for b in "${backbone[@]}"; do
    echo $b
    for t in "${task[@]}"; do
        for s in "${seed[@]}"; do
            CUDA_VISIBLE_DEVICES=$gpu nohup python train_encoder.py --dataset $dataset --task $t --backbone $b --seed $s > /dev/null &
            gpu=$((gpu + 1))
            if [ $gpu -eq $world_size ]; then
                gpu=0
                wait
            fi
        done
    done
done

gpu=0
for b in "${backbone[@]}"; do
    echo $b
    for t in "${task[@]}"; do
        for s in "${seed[@]}"; do
            CUDA_VISIBLE_DEVICES=$gpu nohup python train_intellicare.py --dataset $dataset --suffix $suffix --task $t --backbone $b --seed $s --text_path $text_path --text_num $text_num > /dev/null &
            gpu=$((gpu + 1))
            if [ $gpu -eq $world_size ]; then
                gpu=0
                wait
            fi
        done
    done
done
