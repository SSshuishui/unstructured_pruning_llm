# Import necessary modules
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from collections import defaultdict
from utils.modelutils import *


@torch.no_grad()
def llama_eval(args, model, testenc, dev,  dataset: str, logger):
    logger.info("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    hf_device_map = model.hf_device_map
    logger.info(hf_device_map)

    for i in range(len(layers)):
        logger.info(f'================={i}==================')
        hf_device = "cuda:0" if len(hf_device_map) == 1 and '' in hf_device_map else f"cuda:{hf_device_map[f'model.layers.{i}']}"
        layer = layers[i].to(hf_device)
        inps = inps.to(hf_device)
        position_ids = position_ids.to(hf_device)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][
                    int(W.numel() * args.sparsity_ratio)
                ]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    hf_device = "cuda:0" if len(hf_device_map) == 1 and '' in hf_device_map else f"cuda:{hf_device_map[f'model.layers.{i}']}"
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(hf_device)
    model.lm_head = model.lm_head.to(hf_device)

    testenc = testenc.to(hf_device)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    logger.info(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache


def process_eval_results(results, logger):
    metric_vals = {}
    for task, result in results['results'].items():
        # 定义指标键的优先级
        priority_keys = [
            'acc_norm,none', 'acc_norm',
            'acc,none', 'acc',
            'accuracy'  # 某些任务可能使用全称
        ]
        
        # 查找有效指标
        value = 0.0
        for key in priority_keys:
            if key in result:
                value = result[key]
                break
        
        # 无有效指标的备选方案
        if value == 0.0:
            acc_keys = [k for k in result if k.startswith('acc')]
            if acc_keys:
                value = result[acc_keys[0]]
                logger.warning(
                    f"Task {task} 使用备用指标 `{acc_keys[0]}`. "
                    f"全部可用键: {list(result.keys())}"
                )
            else:
                logger.error(f"Task {task} 无可用精度指标!")
        
        metric_vals[task] = round(value, 4)
    
    # 计算平均（排除完全失败的任务）
    valid_scores = [v for v in metric_vals.values() if v > 0]
    avg_score = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0
    metric_vals['acc_avg'] = avg_score
    
    return metric_vals


def eval_zero_shot(args, model, logger, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0,  add_special_tokens=False): 
    import lm_eval
    from lm_eval import utils as lm_eval_utils
    from lm_eval.models.huggingface import HFLM

    model.to(DEV)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)
    
    ALL_TASKS = ["piqa", "copa", "boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
    task_names = lm_eval_utils.pattern_match(task_list, ALL_TASKS)
    print(f"Matched Tasks: {task_names}")
    if not task_names:
        raise ValueError("No tasks matched. Check task names!")

    results = lm_eval.simple_evaluate(
        model=hflm, 
        tasks=task_names, 
        num_fewshot=num_fewshot,
        batch_size=args.lm_eval_batch_size
    )

    metric_vals = process_eval_results(results, logger)
    logger.info(f"Evaluation Results: {metric_vals}")



# Function to evaluate perplexity (ppl) on a specified model
def eval_ppl(model, tokenizer, dataset, device=torch.device("cuda:0")):
    from utils.datautils import get_loaders
    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=2048, tokenizer=tokenizer
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl = eval_ppl_dataset(model, testloader, 1, device)
        
    return ppl 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_dataset(model, testenc, bs=1, device=None):
    model_seqlen = 2048

    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model_seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model_seqlen):(j * model_seqlen)].to(device)
        inputs = inputs.reshape(j-i, model_seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits #[:, :model.seqlen]

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model_seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model_seqlen))

    return ppl.item()