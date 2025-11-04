#!/bin/bash

# 指定模型路径和其他固定参数
MODEL_PATH="/data/BaseLLMs/llama3-8b"
DATASET="c4"
GRADIENT_PATH="./gradients/llama3/gradients_aggregrate_norm_l1_model_llama3-8b_128_0.pth"

# sparsegpt
for sparsity_type in 2:4 4:8; do
    python llama.py \
        --model $MODEL_PATH \
        --prune_method sparsegpt \
        --dataset $DATASET \
        --sparsity_ratio 0.5 \
        --sparsity_type $sparsity_type \
        --save
done


# wanda
for sparsity_type in 2:4 4:8; do
    python llama.py \
        --model $MODEL_PATH \
        --prune_method wanda \
        --dataset $DATASET \
        --sparsity_ratio 0.5 \
        --sparsity_type $sparsity_type \
        --save
done

# admm
for sparsity_type in 2:4 4:8; do
    python llama.py \
        --model $MODEL_PATH \
        --prune_method admm \
        --dataset $DATASET \
        --sparsity_ratio 0.5 \
        --sparsity_type $sparsity_type \
        --save
done

# RIA
for sparsity_type in 2:4 4:8; do
    python llama.py \
        --model $MODEL_PATH \
        --prune_method RIA \
        --dataset $DATASET \
        --sparsity_ratio 0.5 \
        --sparsity_type $sparsity_type \
        --save
done

# owl
for sparsity_type in 2:4 4:8; do
    python llama.py \
        --model $MODEL_PATH \
        --prune_method owl \
        --initial_method sparsegpt \
        --dataset $DATASET \
        --sparsity_ratio 0.5 \
        --sparsity_type $sparsity_type \
        --Lamda 2 \
        --Hyper_m 6 \
        --save
done


for sparsity_type in 2:4 4:8; do
    python llama.py \
        --model $MODEL_PATH \
        --prune_method owl \
        --initial_method wanda \
        --dataset $DATASET \
        --sparsity_ratio 0.5 \
        --sparsity_type $sparsity_type \
        --Lamda 2 \
        --Hyper_m 6 \
        --save
done


# alphapruning
for sparsity_type in 2:4 4:8; do
    python llama.py \
        --model $MODEL_PATH \
        --prune_method alphapruning \
        --initial_method sparsegpt \
        --dataset $DATASET \
        --sparsity_ratio 0.5 \
        --sparsity_type $sparsity_type \
        --ww_metric alpha_peak \
        --epsilon 0.3 \
        --save
done


for sparsity_type in 2:4 4:8; do
    python llama.py \
        --model $MODEL_PATH \
        --prune_method alphapruning \
        --initial_method wanda \
        --dataset $DATASET \
        --sparsity_ratio 0.5 \
        --sparsity_type $sparsity_type \
        --ww_metric alpha_peak \
        --epsilon 0.3 \
        --save
done


# DSnoT
for sparsity_type in 2:4 4:8; do
    python llama.py \
        --model $MODEL_PATH \
        --prune_method DSnoT \
        --initial_method wanda \
        --dataset $DATASET \
        --sparsity_ratio 0.5 \
        --sparsity_type $sparsity_type \
        --max_cycle_time 50 \
        --update_threshold 0.1 \
        --pow_of_var_regrowing 1 \
        --save
done



# gradient
for sparsity_type in 2:4 4:8; do
    python llama.py \
        --model $MODEL_PATH \
        --prune_method gradient \
        --dataset $DATASET \
        --sparsity_ratio 0.5 \
        --sparsity_type $sparsity_type \
        --gradient_path $GRADIENT_PATH \
        --save
done


# gblm
for sparsity_type in 2:4 4:8; do
    python llama.py \
        --model $MODEL_PATH \
        --prune_method gblm-pruner \
        --dataset $DATASET \
        --sparsity_ratio 0.5 \
        --sparsity_type $sparsity_type \
        --gradient_path $GRADIENT_PATH \
        --save
done


# pruner-zero
for sparsity_type in 2:4 4:8; do
    python llama.py \
        --model $MODEL_PATH \
        --prune_method pruner-zero \
        --dataset $DATASET \
        --sparsity_ratio 0.5 \
        --sparsity_type $sparsity_type \
        --gradient_path $GRADIENT_PATH \
        --save
done
