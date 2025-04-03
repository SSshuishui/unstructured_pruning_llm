#!/bin/bash

# 定义是否使用 CUDA 和剪枝方法
USE_CUDA=7  # 使用 GPU 0
PRUNE_METHOD="wanda"

# 指定模型路径和其他固定参数
MODEL_PATH="/home/wuhaoran/data/BaseLLMs/llama3-8b"
DATASET="c4"

# 遍历不同的 sparsity_type 和 sparsity_ratio 组合
for sparsity_type in "unstructured" "2:4" "4:8"; do
    case $sparsity_type in
        "unstructured")
            # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
            for sparsity_ratio in 0.2 0.3; do
                CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
                    --model $MODEL_PATH \
                    --prune_method $PRUNE_METHOD \
                    --dataset $DATASET \
                    --sparsity_ratio $sparsity_ratio \
                    --sparsity_type $sparsity_type \
                    --eval_zero_shot
            done
            ;;
        # "2:4"|"4:8")
        #     # 当 sparsity_type 是 2:4 或 4:8 时，sparsity_ratio 固定为 0.5
        #     sparsity_ratio=0.5
        #     CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
        #             --model $MODEL_PATH \
        #             --prune_method $PRUNE_METHOD \
        #             --dataset $DATASET \
        #             --sparsity_ratio $sparsity_ratio \
        #             --sparsity_type $sparsity_type \
        #             --eval_zero_shot
        #     ;;
        *)
            echo "Unknown sparsity_type: $sparsity_type"
            ;;
    esac
done