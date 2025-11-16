#!/bin/bash

# 定义是否使用 CUDA 和剪枝方法
USE_CUDA=0,1

# 指定模型路径和其他固定参数
MODEL_PATH="/data/BaseLLMs/llama2-7b-hf"
DATASET="c4"

PRUNE_METHOD="alphapruning"
INITIAL_METHOD="sparsegpt"

# 遍历不同的 sparsity_type 和 sparsity_ratio 组合
for sparsity_type in "unstructured"; do
    case $sparsity_type in
        "unstructured")
            # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
            for sparsity_ratio in 0.2 0.3 0.4; do
                CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
                    --model $MODEL_PATH \
                    --prune_method $PRUNE_METHOD \
                    --initial_method $INITIAL_METHOD \
                    --dataset $DATASET \
                    --sparsity_ratio $sparsity_ratio \
                    --sparsity_type $sparsity_type \
                    --ww_metric alpha_peak \
                    --epsilon 0.3 \
                    --eval_ppl
            done
            ;;
        *)
            echo "Unknown sparsity_type: $sparsity_type"
            ;;
    esac
done


PRUNE_METHOD="alphapruning"
INITIAL_METHOD="wanda"

# 遍历不同的 sparsity_type 和 sparsity_ratio 组合
for sparsity_type in "unstructured"; do
    case $sparsity_type in
        "unstructured")
            # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
            for sparsity_ratio in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
                CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
                    --model $MODEL_PATH \
                    --prune_method $PRUNE_METHOD \
                    --initial_method $INITIAL_METHOD \
                    --dataset $DATASET \
                    --sparsity_ratio $sparsity_ratio \
                    --sparsity_type $sparsity_type \
                    --ww_metric alpha_peak \
                    --epsilon 0.3 \
                    --eval_ppl
            done
            ;;
        *)
            echo "Unknown sparsity_type: $sparsity_type"
            ;;
    esac
done