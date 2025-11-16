#!/bin/bash

USE_CUDA=0

# 指定模型路径和其他固定参数
MODEL_PATH="/data/BaseLLMs/llama2-7b-hf"
DATASET="c4"

CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
    --model $MODEL_PATH \
    --prune_method none \
    --dataset $DATASET \
    --eval_ppl \
    --eval_zeroshot