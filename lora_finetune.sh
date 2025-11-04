#!/bin/bash

# 定义要搜索的目录
SEARCH_DIR="./save_models"

# 找出所有以数字结尾的目录
find "$SEARCH_DIR" -type d -regex ".*[0-9]$" | while read -r model_dir; do
    # 获取模型目录的名称用于结果文件命名
    model_name=$(basename "$model_dir")
    
    echo "=== Processing model: $model_dir ==="
    python lora_finetune_c4.py \
        --model "$model_dir" \
        --save "./results/ppl.txt"
done
