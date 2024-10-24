# unstructured_pruning_llm
LLM Un-Structured Pruning Methods

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

For unstructured 50% sparsity
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