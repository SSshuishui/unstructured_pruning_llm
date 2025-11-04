import torch
import torch.nn as nn
import gc
import time
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer)
from peft import (
    LoraConfig,
    get_peft_model
    )
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='LLaMA model')
parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
parser.add_argument('--save', type=str, default=None, help='Path to save results.')
parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
args = parser.parse_args()

print("loading model...")
start_time = time.time()

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.config.use_cache = False

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    use_fast=False,
    padding_side='right',
    trust_remote_code=True,
    add_eos_token=True,
    add_bos_token=True
    )

tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
        'gate_proj',
        'up_proj',
        'down_proj',
        ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
    )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

train_batch_size=4

original_dataset = 'vicgalle/alpaca-gpt4'
split = f"train[:10000]"
dataset = load_dataset('parquet', data_files={'/data/huggingface_data/alpaca_gpt4_train.parquet'}, split=split)

# Set Supervised Finetuning Trainer (SFTTrainer) parameters
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config,
    args = SFTConfig(
        output_dir= './results',
        num_train_epochs= 1,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=2,
        optim = 'paged_adamw_8bit',
        save_strategy="no",
        logging_steps= 20,
        learning_rate= 2e-4,
        weight_decay= 0.001,
        fp16=False,
        bf16=False,
        max_grad_norm= 0.3,
        max_steps= -1,
        warmup_ratio= 0.3,
        group_by_length= True,
        lr_scheduler_type= 'linear',
        report_to="none",
        max_seq_length=1024,
        dataset_text_field='text',
        packing=False,
    ),
)

# Train model
trainer.train()

print("*"*30)

end_time = time.time()
elapsed_time = (end_time - start_time)/60
print(f"All running time: {elapsed_time:.2f} min")


model.merge_and_unload()

device = torch.device("cuda:0")
from eval import eval_ppl
for dataset in ['wikitext2', 'c4', 'ptb']:
    ppl = eval_ppl(model, tokenizer, dataset, device)
    print(f"\n{args.model}: ppl on {dataset}: {ppl}\n")