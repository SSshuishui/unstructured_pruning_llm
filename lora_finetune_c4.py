import torch
import torch.nn as nn
import os
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
from trl import SFTTrainer, SFTConfig # SFTConfig is part of trl
import argparse

parser = argparse.ArgumentParser()
# Changed help text to be more descriptive for models
parser.add_argument('--model', type=str, required=True, help='Path or Hugging Face ID of the LLM model to finetune (e.g., decapoda-research/llama-7b-hf, meta-llama/Llama-2-7b-hf).')
parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility.')
# Note: The following args are kept from the original script but might not be used by the evaluation part depending on its implementation.
parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples (if needed by eval_ppl).')
parser.add_argument('--save', type=str, default=None, help='Path to save evaluation results (if needed by eval_ppl).')
parser.add_argument('--save_model', type=str, default=None, help='Path to save the finetuned model (optional).') # Renamed slightly for clarity
args = parser.parse_args()

# Set seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

print(f"loading model: {args.model}...")
start_time = time.time()

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16, # Use bfloat16 if your GPU supports it (Ampere/Ada or newer)
    device_map="auto"
)

# model_seqlen is not strictly needed here as max_seq_length is in SFTConfig
# model_seqlen = 2048
model.config.use_cache = False
# Set pad_token_id for models like Llama that don't have one by default
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id
    print(f"Model's pad_token_id not set, using eos_token_id: {model.config.pad_token_id}")


# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    use_fast=False, # Often recommended for Llama
    padding_side='right', # Recommended for causal models
    trust_remote_code=True,
    )

# Set pad token if not already set
if tokenizer.pad_token is None:
     print("Tokenizer pad token not set, using eos_token as pad_token.")
     tokenizer.pad_token = tokenizer.eos_token
elif tokenizer.pad_token != tokenizer.eos_token and tokenizer.pad_token_id == tokenizer.eos_token_id:
     # Handle cases where pad_token is set but maps to the same ID as eos_token
     print(f"Tokenizer pad_token '{tokenizer.pad_token}' maps to same ID as eos_token.")


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

train_batch_size=2

# --- MODIFIED PART: Dataset Loading ---
print("Loading C4 training dataset...")
try:
    # Load the C4 dataset, specifically the 'realnewsv2' config for train split.
    # train_dataset = load_dataset("c4", "realnewsv2", split="train", streaming=False)
    train_dataset = load_dataset(
        'bhxiang/c4_calibrate_mini', split='train'
    )
    print(f"Loaded C4 training dataset with {len(train_dataset)} samples.")

    # Optional: Filter out empty or None text samples from C4, as they can cause issues
    print("Filtering empty or None text samples from dataset...")
    original_len = len(train_dataset)
    train_dataset = train_dataset.filter(lambda x: x['text'] is not None and len(x['text'].strip()) > 0)
    print(f"Filtered {original_len - len(train_dataset)} samples. Remaining: {len(train_dataset)}")

    # Optional: Take a smaller subset for faster testing/debugging
    train_dataset = train_dataset.select(range(10000)) # Use first 50,000 samples for testing
    print(f"Using training dataset with {len(train_dataset)} samples for SFTTrainer.")


except Exception as e:
    print(f"Error loading C4 dataset: {e}")
    print("Please ensure you have internet access and sufficient disk space to download the C4 dataset.")
    exit() # Exit if dataset loading fails

# Set Supervised Finetuning Trainer (SFTTrainer) parameters
print("Setting up SFTTrainer for C4 dataset...")

# Your original SFTConfig parameters are mostly fine for C4 as well.
# We just need to ensure the dataset_text_field is correctly pointing to the 'text' column.
trainer_args = SFTConfig(
    output_dir="./results", # Specify an output directory for logs
    num_train_epochs= 1,
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps=4,
    optim = 'paged_adamw_8bit',
    save_strategy="no", # Keep saving strategy as no
    logging_steps= 20,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16=False, # Set True if your GPU and setup support fp16, often faster/less VRAM
    bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8, # Use bf16 on modern GPUs
    max_grad_norm= 0.3,
    max_steps= -1, # -1 means train for num_train_epochs
    warmup_ratio= 0.03, # A lower warmup ratio like 0.03 is more common
    group_by_length= True, # Group samples by length for efficiency
    lr_scheduler_type= 'linear',
    report_to="none", # Disable reporting unless configured
    max_seq_length=1024, # Maximum sequence length for training samples
    dataset_text_field='text', # *** THIS IS CRUCIAL FOR C4 *** Pointing to the 'text' column
    packing=False, # Set to True to pack multiple short samples into one sequence (more efficient, slightly different training)
    evaluation_strategy="no",
    do_eval=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset, # Pass the loaded C4 training dataset
    peft_config=peft_config, # Pass the LoRA config
    args = trainer_args, # Pass the SFTConfig/TrainingArguments
)


# Train model
print("Starting training...")
train_start_time = time.time()
trainer.train()
train_end_time = time.time()
train_elapsed_time = (train_end_time - train_start_time)/60
print(f"Training finished in: {train_elapsed_time:.2f} min")

# --- Save Finetuned Model (Optional) ---
if args.save_model:
    print(f"Saving finetuned LoRA adapters to {args.save_model}...")
    # To save only the LoRA adapters
    trainer.model.save_pretrained(args.save_model)
    # merged_model = model.merge_and_unload()
    # merged_model.save_pretrained(args.save_model)
    # tokenizer.save_pretrained(args.save_model) # Always save tokenizer with the model
    print("LoRA adapters saved.")


print("\n" + "*"*30)
print("Training complete. Merging LoRA and starting evaluation.")
print("*"*30 + "\n")


# Merge LoRA adapters before evaluation
print("Merging LoRA adapters for evaluation...")
model = model.merge_and_unload()
print("LoRA adapters merged.")

# Ensure model is in evaluation mode and potentially moved to a single device
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assume eval_ppl function is available from eval.py
try:
    from eval import eval_ppl # Import the evaluation function
    print("Successfully imported eval_ppl function.")
except ImportError:
    print("Error: Could not import 'eval_ppl' from 'eval.py'.")
    print("Please make sure you have an 'eval.py' script in the same directory")
    print("that defines and exports the 'eval_ppl' function with signature `eval_ppl(model, tokenizer, dataset_name, device)`.")
    exit() # Exit if eval_ppl is not available


# Evaluate PPL on specified datasets (including C4 validation)
print("Starting Perplexity (PPL) Evaluation")
evaluation_datasets = ['wikitext2', 'c4', 'ptb'] # Evaluate on C4 validation set

ppl_results = {}
for dataset_name in evaluation_datasets:
    print(f"\nEvaluating PPL on {dataset_name} validation set...")
    eval_start_time = time.time()
    try:
        # The eval_ppl function is expected to load the *validation* split of dataset_name
        ppl = eval_ppl(model, tokenizer, dataset_name, device)
        eval_end_time = time.time()
        eval_elapsed_time = (eval_end_time - eval_start_time) / 60

        ppl_results[dataset_name] = ppl
        print(f"\n{args.model}: PPL on {dataset_name} validation set: {ppl:.4f}")
        print(f"Evaluation on {dataset_name} took: {eval_elapsed_time:.2f} min")
    except Exception as e:
        print(f"An error occurred during evaluation on {dataset_name}: {e}")
        import traceback
        traceback.print_exc()

if args.save:
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        with open(args.save, 'a') as f:
            f.write(f"Model: {args.model}\n")
            f.write("-" * 20 + "\n")
            f.write("Perplexity (PPL) Results:\n")
            for dataset_name, ppl_value in ppl_results.items():
                if isinstance(ppl_value, float):
                     f.write(f"{dataset_name}: {ppl_value:.4f}\n")
                else:
                     f.write(f"{dataset_name}: {ppl_value}\n") # Write error message if evaluation failed
        print(f"\nPerplexity results saved to {args.save}")
    except Exception as e:
        print(f"\nError saving perplexity results to {args.save}: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n--save argument not provided. PPL results not saved to a file.")


print("\n" + "*"*30)
print("Finetuning and Evaluation complete.")
print("*"*30)

end_time = time.time()
elapsed_time = (end_time - start_time) / 60
print(f"Total script running time: {elapsed_time:.2f} min")
