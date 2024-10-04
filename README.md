# unstructured_pruning_llm
LLM Un-Structured Pruning Methods

## Magnitude baseline
```
python llama.py \
    --model /PATH/TO/LLAMA3/ \
    --prune_method magnitude \
    --dataset c4 \
    --sparsity_ratio .5 \
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
    --save_mdoel save_models/sparsegpt/
```
#### Prune to full 2:4 sparsity
```
python llama.py \
    --model /PATH/TO/LLAMA3/ \
    --prune_method sparsegpt \
    --dataset c4 \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save_mdoel save_models/sparsegpt/
```
#### Prune to 50\% + 4-bit
```
python llama.py \
    --model /PATH/TO/LLAMA3/ \
    --prune_method sparsegpt \
    --dataset c4 \
    --sparsity_ratio 0.5 \
    --wbits 4 \
    --save_mdoel save_models/sparsegpt/
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
    --save_mdoel save_models/wanda/ 
```

For unstructured 50% sparsity
```
python llama.py \
    --model /PATH/TO/LLAMA3/ \
    --prune_method wanda \
    --dataset c4 \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save_mdoel save_models/wanda/ 
```