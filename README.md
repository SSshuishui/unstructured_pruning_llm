# unstructured_pruning_llm
LLM Un-Structured Pruning Methods

Include:
| Methods     | Quantize | PPL Eval | Task Eval | Save |
| :--------   | :-------:| :------: | :-------: | :---:|
| Magnitude   | ✅ | ✅ | TODO | TODO 
| SparseGPT   | ✅ | ✅ | TODO | TODO 
| Wanda       | ✅ | ✅ | TODO | TODO 
| SparseLLM   | ✅ | ✅ | TODO | TODO 
| DSnoT       | ✅ | ✅ | TODO | TODO 
| OWL         | ✅ | ✅ | TODO | TODO 
| GBLM-Pruner | ✅ | ✅ | TODO | TODO 
| Pruner-Zero | ✅ | ✅ | TODO | TODO 
| FLAP        | ✅ | ✅ | TODO | TODO 
| admm        | TODO | TODO | TODO | TODO 
| RIA         | TODO | TODO | TODO | TODO 
| rethinking llm prune | TODO | TODO | TODO | TODO 



add `--eval_zero_shot` to evaluate 
```
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
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
--gmp
```


## SparseGPT
#### Prune to 50\% uniform sparsity
```
python llama.py \
--model /PATH/TO/LLAMA3/ \
--prune_method sparsegpt \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--save_model save_models/sparsegpt/
```
#### Prune to full 2:4 sparsity
```
python llama.py \
--model /PATH/TO/LLAMA3/ \
--prune_method sparsegpt \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--save_model save_models/sparsegpt/
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
--save_model save_models/sparsegpt/
```


## Wanda
For structured N:M sparsity, "2:4" or "4:8"
```
python llama.py \
--model /PATH/TO/LLAMA3/ \
--prune_method wanda \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--save_model save_models/wanda/ 
```

For unstructured 50% sparsity \
add `--use_variant` for wanda variant
```
python llama.py \
--model /PATH/TO/LLAMA3/ \
--prune_method wanda \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--save_model save_models/wanda/ 
```

## SparseLLM
* Due to 4090 only have 24G, need to minimize nsamples for running.
For unstructured sparsity
```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method sparsellm \
--dataset c4 \
--sparsity_ratio 0.7 \
--sparsity_type unstructured \
--save_model save_models/sparsellm/ \
--nsamples 16
```

For structured N:M sparsity, "2:4" or "4:8"
```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method sparsellm \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--save_model save_models/sparsellm/ \
--nsamples 16
```


## DSnoT
For unstructured sparsity
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
--pow_of_var_regrowing 1
```

For structured N:M sparsity, "2:4" or "4:8"
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
--pow_of_var_regrowing 1
```

## OWL
For OWL-Magnitude
```
python  llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method owl \
--dataset c4 \
--initial_method magnitude \
--sparsity_ratio 0.7 \
--sparsity_type unstructured \
--Lamda 0.08 \
--Hyper_m 5 \
--save_model save_models/OWL/magnitude/ 
```

For OWL-SparseGPT
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
--save_model save_models/OWL/sparsegpt/ \
--nsamples 16
```

For OWL-Wanda \
add `--use_variant` for wanda variant
```
python  llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method owl \
--dataset c4 \
--initial_method wanda \
--sparsity_ratio 0.7 \
--sparsity_type unstructured \
--Lamda 0.08 \
--Hyper_m 5 \
--save_model save_models/OWL/wanda/ 
```
```
python  llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method owl \
--dataset c4 \
--initial_method wanda \
--sparsity_ratio 0.7 \
--sparsity_type "4:8" \
--Lamda 0.08 \
--Hyper_m 5 \
--save_model save_models/OWL/wanda/ 
```


## GBLM-Pruner
1. Computate of gradient magnitude for calculation of pruning metric
```
python gradient_computation.py \
--nsamples 128 \
--model /PATH/TO/LLAMA2/ \
--llama_version 2 \
--task gradient
```
2. For unstructured pruning \
add `--use_variant` for wanda variant
```
python  llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method gblm-pruner \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--save_model save_models/gblm_pruner/ \
--gradient_path ./gradients/llama2/gradients_aggregrate_norm_l1_model_llama2-7b-hf_128_0.pth \
--nsamples 16
```
For structured N:M sparsity, "2:4" or "4:8"
```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method gblm-pruner \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--save_model save_models/gblm_pruner/ \
--gradient_path ./gradients/llama2/gradients_aggregrate_norm_l1_model_llama2-7b-hf_128_0.pth \
--nsamples 16
```


## Pruner-Zero

1. Computate of gradient magnitude for calculation of pruning metric
```
python gradient_computation.py 
--nsamples 128 \
--model /PATH/TO/LLAMA2/ \
--llama_version 2 \
--task gradient
```
2. For unstructured pruning \
add `--use_variant` for wanda variant
```
python  llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method pruner-zero \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--save_model save_models/pruner_zero/ \
--gradient_path ./gradients/llama2/gradients_aggregrate_norm_l1_model_llama2-7b-hf_128_0.pth \
--nsamples 16 
```
For structured N:M sparsity, "2:4" or "4:8"
```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method pruner-zero \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--save_model save_models/pruner_zero/ \
--gradient_path ./gradients/llama2/gradients_aggregrate_norm_l1_model_llama2-7b-hf_128_0.pth \
--nsamples 16
```


## FLAP
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method flap \
--dataset c4 \
--sparsity_ratio 0.2 \
--remove_heads -1 \
--metrics WIFV \
--structure AL-AM \
--nsamples 1024 \
--save_model save_models/flap/ \



## Admm
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method admm \
--dataset c4 \
--sparsity_ratio 0.6 \
--sparsity_type unstructured \
--save_model save_models/admm/ 


For structured N:M sparsity, "2:4" or "4:8"
```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method admm \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--save_model save_models/admm/ \
```


## RIA (Relative Importance and Activations)
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method RIA \
--dataset c4 \
--sparsity_ratio 0.6 \
--sparsity_type unstructured \
--save_model save_models/RIA/ \


For structured N:M sparsity, "2:4" or "4:8"
```
python llama.py \
--model /PATH/TO/LLAMA2/ \
--prune_method RIA \
--dataset c4 \
--sparsity_ratio 0.5 \
--sparsity_type 2:4 \
--save_model save_models/RIA/ \
```