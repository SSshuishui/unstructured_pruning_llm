#!/bin/bash

# 定义是否使用 CUDA 和剪枝方法
USE_CUDA=0,1,2  # 使用 GPU 0
PRUNE_METHOD="wanda"

# 指定模型路径和其他固定参数
MODEL_PATHS=(
    # "/data/BaseLLMs/llama3-8b"
    "/data/BaseLLMs/llama2-7b-hf"
)
DATASET="c4"

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "Evaluating model at: $MODEL_PATH"

    # PRUNE_METHOD="wanda"
    # for sparsity_type in "unstructured"; do
    #     case $sparsity_type in
    #         "unstructured")
    #             # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
    #             for sparsity_ratio in 0.7 0.8; do
    #                 CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
    #                     --model $MODEL_PATH \
    #                     --prune_method $PRUNE_METHOD \
    #                     --dataset $DATASET \
    #                     --sparsity_ratio $sparsity_ratio \
    #                     --sparsity_type $sparsity_type \
    #                     --eval_ppl
    #             done
    #             ;;
    #         *)
    #             echo "Unknown sparsity_type: $sparsity_type"
    #             ;;
    #     esac
    # done

#     PRUNE_METHOD="alphapruning"
#     INITIAL_METHOD="wanda"

#     for sparsity_type in "unstructured"; do
#         case $sparsity_type in
#             "unstructured")
#                 # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
#                 for sparsity_ratio in 0.5 0.6 0.7 0.8; do
#                     CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
#                         --model $MODEL_PATH \
#                         --prune_method $PRUNE_METHOD \
#                         --initial_method $INITIAL_METHOD \
#                         --dataset $DATASET \
#                         --sparsity_ratio $sparsity_ratio \
#                         --sparsity_type $sparsity_type \
#                         --ww_metric alpha_peak \
#                         --epsilon 0.3 \
#                         --eval_ppl
#                 done
#                 ;;
#             *)
#                 echo "Unknown sparsity_type: $sparsity_type"
#                 ;;
#         esac
#     done


    PRUNE_METHOD="owl"
    INITIAL_METHOD="wanda"

    for sparsity_type in "unstructured"; do
        case $sparsity_type in
            "unstructured")
                # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
                for sparsity_ratio in 0.7 0.8; do
                    CUDA_VISIBLE_DEVICES=$USE_CUDA python llama2.py \
                        --model $MODEL_PATH \
                        --prune_method $PRUNE_METHOD \
                        --initial_method $INITIAL_METHOD \
                        --dataset $DATASET \
                        --sparsity_ratio $sparsity_ratio \
                        --sparsity_type $sparsity_type \
                        --Lamda 0.08 \
                        --Hyper_m 5 \
                        --eval_ppl
                done
                ;;
            *)
                echo "Unknown sparsity_type: $sparsity_type"
                ;;
        esac
    done


#     PRUNE_METHOD="DLP"
#     INITIAL_METHOD="wanda"
#     for sparsity_type in "unstructured"; do
#         case $sparsity_type in
#             "unstructured")
#                 # 当 sparsity_type 是 unstructured 时，遍历 sparsity_ratio
#                 for sparsity_ratio in 0.5 0.6 0.7 0.8; do
#                     CUDA_VISIBLE_DEVICES=$USE_CUDA python llama.py \
#                         --model $MODEL_PATH \
#                         --prune_method $PRUNE_METHOD \
#                         --initial_method $INITIAL_METHOD \
#                         --dataset $DATASET \
#                         --sparsity_ratio $sparsity_ratio \
#                         --sparsity_type $sparsity_type \
#                         --dlp_alpha 0.15 \
#                         --eval_ppl
#                 done
#                 ;;
#             *)
#                 echo "Unknown sparsity_type: $sparsity_type"
#                 ;;
#         esac
#     done

done