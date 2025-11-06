#!/bin/bash

USE_CUDA=0,1 # 使用 GPU 0
PRUNE_METHOD="sparsegpt"

# 指定模型路径和其他固定参数
MODEL_PATH="/data/BaseLLMs/llama3-8b"
DATASET="c4"

# 遍历不同的 sparsity_type 和 sparsity_ratio 组合
for sparsity_type in "unstructured"; do
    case $sparsity_type in
        "unstructured")
            # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
            for sparsity_ratio in 0.5 0.6 0.7 0.8; do
            # for sparsity_ratio in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
                CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
                    --model $MODEL_PATH \
                    --prune_method $PRUNE_METHOD \
                    --dataset $DATASET \
                    --sparsity_ratio $sparsity_ratio \
                    --sparsity_type $sparsity_type \
                    --mar \
                    --eval_ppl
            done
            ;;
        *)
            echo "Unknown sparsity_type: $sparsity_type"
            ;;
    esac
done

for sparsity_type in "unstructured"; do
    case $sparsity_type in
        "unstructured")
            # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
            for sparsity_ratio in 0.5 0.6 0.7 0.8; do
            # for sparsity_ratio in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
                CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
                    --model $MODEL_PATH \
                    --prune_method $PRUNE_METHOD \
                    --dataset $DATASET \
                    --sparsity_ratio $sparsity_ratio \
                    --sparsity_type $sparsity_type \
                    --mar \
                    --groupsize 128 \
                    --eval_ppl 
            done
            ;;
        *)
            echo "Unknown sparsity_type: $sparsity_type"
            ;;
    esac
done



MODEL_PATH="/data/BaseLLMs/llama2-7b-hf"

# 遍历不同的 sparsity_type 和 sparsity_ratio 组合
for sparsity_type in "unstructured"; do
    case $sparsity_type in
        "unstructured")
            # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
            for sparsity_ratio in 0.5 0.6 0.7 0.8; do
            # for sparsity_ratio in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
                CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
                    --model $MODEL_PATH \
                    --prune_method $PRUNE_METHOD \
                    --dataset $DATASET \
                    --sparsity_ratio $sparsity_ratio \
                    --sparsity_type $sparsity_type \
                    --mar \
                    --eval_ppl 
            done
            ;;
        *)
            echo "Unknown sparsity_type: $sparsity_type"
            ;;
    esac
done


for sparsity_type in "unstructured"; do
    case $sparsity_type in
        "unstructured")
            # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
            for sparsity_ratio in 0.5 0.6 0.7 0.8; do
            # for sparsity_ratio in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
                CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
                    --model $MODEL_PATH \
                    --prune_method $PRUNE_METHOD \
                    --dataset $DATASET \
                    --sparsity_ratio $sparsity_ratio \
                    --sparsity_type $sparsity_type \
                    --mar \
                    --groupsize 128 \
                    --eval_ppl 
            done
            ;;
        *)
            echo "Unknown sparsity_type: $sparsity_type"
            ;;
    esac
done