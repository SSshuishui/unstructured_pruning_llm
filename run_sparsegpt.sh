#!/bin/bash

# 定义是否使用 CUDA 和剪枝方法
USE_CUDA=2 
PRUNE_METHOD="sparsegpt"

# 指定模型路径和其他固定参数
MODEL_PATH="/data/BaseLLMs/llama2-7b-hf"
DATASET="c4"

# 遍历不同的 sparsity_type 和 sparsity_ratio 组合
for sparsity_type in "unstructured"; do
    case $sparsity_type in
        "unstructured")
            # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
            for sparsity_ratio in 0.2; do
                CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
                    --model $MODEL_PATH \
                    --prune_method $PRUNE_METHOD \
                    --dataset $DATASET \
                    --sparsity_ratio $sparsity_ratio \
                    --sparsity_type $sparsity_type \
                    --eval_zeroshot
            done
            ;;
        *)
            echo "Unknown sparsity_type: $sparsity_type"
            ;;
    esac
done