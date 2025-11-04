# unstructured_pruning_llm

LLM Un-Structured Pruning Methods

Include:

| Methods                                          | Prune | PPL Eval | Task Eval | Save |
| :----------------------------------------------- | :------: | :------: | :-------: | :--: |
| Magnitude                                        |   yes   |   yes   |   yes   | yes |
| [SparseGPT](https://arxiv.org/pdf/2301.00774)       |   yes   |   yes   |   yes   | yes |
| [Wanda](https://arxiv.org/pdf/2306.11695)           |   yes   |   yes   |   yes   | yes |
| [DSnoT](https://arxiv.org/pdf/2310.08915)           |   yes   |   yes   |   yes   | yes |
| [OWL](https://arxiv.org/pdf/2310.05175)             |   yes   |   yes   |   yes   | yes |
| [GBLM-Pruner](https://arxiv.org/pdf/2311.04902)     |   yes   |   yes   |   yes   | yes |
| [Pruner-Zero](https://arxiv.org/pdf/2406.02924v1)   |   yes   |   yes   |   yes   | yes |
| [FLAP](https://arxiv.org/pdf/2312.11983)            |   yes   |   yes   |   yes   | yes |
| [admm](https://arxiv.org/pdf/2401.02938)            |   yes   |   yes   |   yes   | yes |
| [RIA](https://openreview.net/forum?id=Tr0lPx9woF)   |   yes   |   yes   |   yes   | yes |
| [AlphaPruninig](https://arxiv.org/pdf/2410.10912)   |   yes   |   yes   |   yes   | yes |
| [ALPS](https://arxiv.org/pdf/2406.07831)            |   yes   |   yes   |   yes   | yes |
| [DLP](https://arxiv.org/pdf/2505.23807)             |   yes   |   yes   |   yes   | yes |

add `--eval_zero_shot` to evaluate

```
cd lm-evaluation-harness
pip install -e .
```

## Magnitude baseline

```
python llama.py \
--model /PATH/TO/LLAMA3/ \
--prune_method magnitude \
--dataset c4 \
--sparsity_ratio .5 \
--sparsity_type unstructured \
--gmp \
--save
```

## SparseGPT

#### unstructured

```
python llama.py \
--model /PATH/TO/LLAMA3/ \
--prune_method sparsegpt \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--save 
```

#### Prune to full 2:4 sparsity

```
python llama.py \
--model /PATH/TO/LLAMA3/ \
--prune_method sparsegpt \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--save
```

#### Prune to 50\% + 4-bit

```
python llama.py \
--model /PATH/TO/LLAMA3/ \
--prune_method sparsegpt \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--wbits 4 \
--save
```

## Wanda
#### unstructured

```
python llama.py \
--model /PATH/TO/LLAMA3/ \
--prune_method wanda \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--save 
```

#### For structured N:M sparsity, "2:4" or "4:8"

```
python llama.py \
--model /PATH/TO/LLAMA3/ \
--prune_method wanda \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--save 
```

#### For unstructured 50% sparsity

add `--use_variant` for wanda variant

```
python llama.py \
--model /PATH/TO/LLAMA3/ \
--prune_method wanda \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--save 
```

## DSnoT

* Due to 4090 only have 24G, need to minimize nsamples for running.

#### For unstructured sparsity 
##### initial_method: sparsegpt | wanda

```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method DSnoT \
--dataset c4 \
--initial_method wanda \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--max_cycle_time 50 \
--update_threshold 0.1 \
--pow_of_var_regrowing 1 \
--save
```

#### For structured N:M sparsity, "2:4" or "4:8"

```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method DSnoT \
--dataset c4 \
--initial_method wanda \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--max_cycle_time 50 \
--update_threshold 0.1 \
--pow_of_var_regrowing 1 \
--save
```

## OWL

* Due to 4090 only have 24G, need to minimize nsamples for running.

#### For OWL-SparseGPT

```
python  llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method owl \
--dataset c4 \
--initial_method sparsegpt \
--sparsity_ratio 0.7 \
--sparsity_type unstructured \
--Lamda 0.08 \
--Hyper_m 5 \
--nsamples 16 \
--save
```

#### For OWL-Wanda

```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method owl \
--dataset c4 \
--initial_method wanda \
--sparsity_ratio 0.7 \
--sparsity_type unstructured \
--Lamda 0.08 \
--Hyper_m 5 \
--save
```

```
python  llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method owl \
--dataset c4 \
--initial_method wanda \
--sparsity_ratio 0.5 \
--sparsity_type "4:8" \
--Lamda 0.08 \
--Hyper_m 5 \
--save
```

## Gradient

#### 1. Computate of gradient magnitude for calculation of pruning metric

```
python gradient_computation.py \
--nsamples 128 \
--model /PATH/TO/LLAMA2/ \
--llama_version 2 \
--task gradient
```

#### 2. For unstructured pruning

* Due to 4090 only have 24G, need to minimize nsamples for running.

  add `--use_variant` for wanda variant

```
python  llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method gradient \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--gradient_path ./gradients/llama2/gradients_aggregrate_norm_l1_model_llama2-7b-hf_128_0.pth \
--nsamples 16 \
--save 
```

#### For structured N:M sparsity, "2:4" or "4:8"

```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method gradient \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--gradient_path ./gradients/llama2/gradients_aggregrate_norm_l1_model_llama2-7b-hf_128_0.pth \
--nsamples 16 \
--save 
```

## GBLM-Pruner

#### 1. Computate of gradient magnitude for calculation of pruning metric

```
python gradient_computation.py \
--nsamples 128 \
--model /PATH/TO/LLAMA2/ \
--llama_version 2 \
--task gradient
```

#### 2. For unstructured pruning

* Due to 4090 only have 24G, need to minimize nsamples for running.

  add `--use_variant` for wanda variant

```
python  llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method gblm-pruner \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--gradient_path ./gradients/llama2/gradients_aggregrate_norm_l1_model_llama2-7b-hf_128_0.pth \
--nsamples 16 \
--save 
```

#### For structured N:M sparsity, "2:4" or "4:8"

```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method gblm-pruner \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--gradient_path ./gradients/llama2/gradients_aggregrate_norm_l1_model_llama2-7b-hf_128_0.pth \
--nsamples 16 \
--save 
```

## Pruner-Zero

#### 1.Computate of gradient magnitude for calculation of pruning metric

```
python gradient_computation.py 
--nsamples 128 \
--model /PATH/TO/LLAMA2/ \
--llama_version 2 \
--task gradient
```

#### 2.For unstructured pruning

* Due to 4090 only have 24G, need to minimize nsamples for running.

  add `--use_variant` for wanda variant

```
python  llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method pruner-zero \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--gradient_path ./gradients/llama2/gradients_aggregrate_norm_l1_model_llama2-7b-hf_128_0.pth \
--nsamples 16 \
--save 
```

#### For structured N:M sparsity, "2:4" or "4:8"

```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method pruner-zero \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--gradient_path ./gradients/llama2/gradients_aggregrate_norm_l1_model_llama2-7b-hf_128_0.pth \
--nsamples 16 \
--save  
```

## FLAP

#### For unstructured sparsity

```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method flap \
--dataset c4 \
--sparsity_ratio 0.2 \
--sparsity_type unstructured \
--remove_heads -1 \
--metrics WIFV \
--structure AL-AM \
--nsamples 1024 \
--save 
```

## Admm

#### For unstructured sparsity

```
python llama.py  \
--model /PATH/TO/LLAMA2/  \
--prune_method admm  \
--dataset c4  \
--sparsity_ratio 0.6  \
--sparsity_type unstructured  \
--save 
```

#### For structured N:M sparsity, "2:4" or "4:8"

```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method admm \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--save 
```

## RIA (Relative Importance and Activations)

#### For unstructured sparsity

```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method RIA \
--dataset c4 \
--sparsity_ratio 0.6 \
--sparsity_type unstructured \
--save 
```

#### For structured N:M sparsity, "2:4" or "4:8"

```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method RIA \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--save 
```

## AlphaPruning

#### unstructured
##### initial_method: sparsegpt | wanda
```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method alphapruning \
--initial_method wanda \
--dataset c4 \
--sparsity_ratio 0.6 \
--sparsity_type unstructured \
--ww_metric alpha_peak \
--epsilon 0.3 \
--save
```

#### structured
```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method alphapruning \
--initial_method wanda \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--ww_metric alpha_peak \
--epsilon 0.3 \
--save
```



## ALPS

#### For unstructured sparsity

* Due to 4090 only have 24G, need to minimize nsamples for running.

```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method ALPS \
--dataset c4 \
--sparsity_ratio 0.6 \
--sparsity_type unstructured \
--save 
```

#### For structured N:M sparsity, "2:4" or "4:8"

```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method ALPS \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--save 
```

## DLP
#### For DLP-SparseGPT

```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method DLP \
--dataset c4 \
--initial_method sparsegpt \
--dlp_alpha 0.15 \
--sparsity_ratio 0.7 \
--sparsity_type unstructured \
--save 
```

#### For OWL-Wanda
```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method DLP \
--dataset c4 \
--initial_method wanda \
--dlp_alpha 0.15 \
--sparsity_ratio 0.7 \
--sparsity_type unstructured \
--save 
```



## Inference Speed Evaluation
### LLaMA2 Model
 - 1. save pruned model(`SOURCE_PATH`) to `SAVE_PATH`, copy the tokenizer file(`tokenizer_config.json`, `tokenizer.json`, `special_tokens_map.json`) to `SOURCE_PATH`
 - 2. install [sparseml](https://github.com/neuralmagic/sparseml) and [deepsparse](https://github.com/neuralmagic/deepsparse) 
 - 3. export pruned model to ONNX format

 ```
 sparseml.export 
 --target_path SAVE_PATH  \
 --integration transformers \
 --task text-generation  \
 SOURCE_PATH
 ```
 - 4.deepsparse inference evaluation
 ```
 deepsparse.benchmark
 SAVE_PATH/deployment/model.onnx \
 --sequence_length 2048 \
 ```

### LLaMA3 Model
 - 1. save pruned model(`SOURCE_PATH`) to `SAVE_PATH`, copy the tokenizer file(`tokenizer_config.json`, `tokenizer.json`, `special_tokens_map.json`) to `SOURCE_PATH`
 - 2. install [vllm](https://github.com/vllm-project/vllm)
 - 3. run benchmark script
 ```
 vllm serve <your_model> --disable-log-requests

 python3 vllm_benchmarks/benchmark_serving.py --backend vllm --model <your_model> --dataset-name <dataset_name> --dataset-path <dataset_path> --num-prompts <number_of_prompts>

 ```

 ```
 python3 vllm_benchmarks/benchmark_throughput.py --model <your_model> --dataset-name <dataset_name> --dataset-path <dataset_path> --num-prompts <number_of_prompts>

 ```


 vllm serve save_models/wanda/unstructured/5 --disable-log-requests

 python vllm_benchmarks/benchmark_serving.py --model save_models/wanda/unstructured/5 --dataset-name sharegpt --dataset-path /data/zhaox/data/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10