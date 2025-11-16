#!/bin/bash

USE_CUDA=0,1 

# 指定模型路径和其他固定参数
MODEL_PATHS=("/data/BaseLLMs/llama3-8b")
DATASET="c4"

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "Evaluating model at: $MODEL_PATH"

    PRUNE_METHOD="mar"
    # 遍历不同的 sparsity_type 和 sparsity_ratio 组合
    for sparsity_type in "unstructured"; do
        case $sparsity_type in
            "unstructured")
                # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
                for sparsity_ratio in 0.7 0.8; do
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


    # PRUNE_METHOD="alphapruning"
    # INITIAL_METHOD="sparsegpt"

    # for sparsity_type in "unstructured"; do
    #     case $sparsity_type in
    #         "unstructured")
    #             # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
    #             for sparsity_ratio in 0.7 0.8; do
    #                 CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
    #                     --model $MODEL_PATH \
    #                     --prune_method $PRUNE_METHOD \
    #                     --initial_method $INITIAL_METHOD \
    #                     --dataset $DATASET \
    #                     --sparsity_ratio $sparsity_ratio \
    #                     --sparsity_type $sparsity_type \
    #                     --ww_metric alpha_peak \
    #                     --epsilon 0.3 \
    #                     --eval_ppl
    #             done
    #             ;;
    #         *)
    #             echo "Unknown sparsity_type: $sparsity_type"
    #             ;;
    #     esac
    # done


    # PRUNE_METHOD="owl"
    # INITIAL_METHOD="sparsegpt"

    # for sparsity_type in "unstructured"; do
    #     case $sparsity_type in
    #         "unstructured")
    #             # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
    #             for sparsity_ratio in 0.7 0.8; do
    #                 CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
    #                     --model $MODEL_PATH \
    #                     --prune_method $PRUNE_METHOD \
    #                     --initial_method $INITIAL_METHOD \
    #                     --dataset $DATASET \
    #                     --sparsity_ratio $sparsity_ratio \
    #                     --sparsity_type $sparsity_type \
    #                     --Lamda 0.08 \
    #                     --Hyper_m 5 \
    #                     --eval_ppl
    #             done
    #             ;;
    #         *)
    #             echo "Unknown sparsity_type: $sparsity_type"
    #             ;;
    #     esac
    # done


    # PRUNE_METHOD="DLP"
    # INITIAL_METHOD="sparsegpt"

    # for sparsity_type in "unstructured"; do
    #     case $sparsity_type in
    #         "unstructured")
    #             # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
    #             for sparsity_ratio in 0.7 0.8; do
    #                 CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
    #                     --model $MODEL_PATH \
    #                     --prune_method $PRUNE_METHOD \
    #                     --initial_method $INITIAL_METHOD \
    #                     --dataset $DATASET \
    #                     --sparsity_ratio $sparsity_ratio \
    #                     --sparsity_type $sparsity_type \
    #                     --dlp_alpha 0.15 \
    #                     --eval_ppl
    #             done
    #             ;;
    #         *)
    #             echo "Unknown sparsity_type: $sparsity_type"
    #             ;;
    #     esac
    # done

done