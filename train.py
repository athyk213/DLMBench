#    Modification Copyright 2024 Jiajun Zhu
#    Modification Copyright 2024 Zhenyu He
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
import torch.nn.functional as F
import builtins
import logging
import os
import math
import glob
import random
import csv
from itertools import chain
from dataclasses import dataclass, field
from typing import Optional

# import wandb
import transformers
from transformers import Trainer, AutoTokenizer, AutoConfig, AutoModel, PreTrainedTokenizerBase
from datasets import load_dataset, load_from_disk
try:
    from models import *
except ImportError:
    pass

from typing import Dict, List, Union
from torch.serialization import add_safe_globals
from deepspeed.runtime.fp16.loss_scaler import LossScaler

add_safe_globals([LossScaler])


CPU_COUNT = os.cpu_count()


@dataclass
class ModelArguments:
    config: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    dataset_cache_dir: str = field(default=None, metadata={"help": "Path to the data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    context_len: int = field(
        default=2048,
        metadata={"help": "Training Context Length."},
    )
    resume_from_checkpoint: Optional[bool] = field(default=None)
    finetune_from_pretrained: Optional[str] = field(default=None)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def get_streaming_dataset(tokenizer, data_args, training_args, cached='tokenized'):
    dpt = data_args.dataset_cache_dir

    if os.path.exists(dpt):
        print(f"*** Loading dataset from {dpt} ***")
        train_raw_dataset = load_from_disk(os.path.join(dpt, 'train'))
        val_raw_dataset = load_from_disk(os.path.join(dpt, 'validation'))
    else:
        print(f"*** Loading streaming dataset from DKYoon/SlimPajama-6B ***")
        train_raw_dataset = load_dataset("DKYoon/SlimPajama-6B", split="train", streaming=True)
        val_raw_dataset = load_dataset("DKYoon/SlimPajama-6B", split='validation', streaming=True)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=training_args.context_len)

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        if total_length >= training_args.context_len:
            total_length = (total_length // training_args.context_len) * training_args.context_len
        result = {
            k: [t[i : i + training_args.context_len] for i in range(0, total_length, training_args.context_len)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    if cached == 'raw':
        return {"train": train_raw_dataset, "validation": val_raw_dataset}
    
    tokenized_datasets = train_raw_dataset.map(tokenize_function, batched=True, remove_columns=["text", "meta", "redpajama_set_name"])
    lm_datasets = {"train": tokenized_datasets.map(group_texts, batched=True), 
                   "validation": val_raw_dataset.map(tokenize_function, batched=True, remove_columns=["text", "meta", "redpajama_set_name"]).map(group_texts, batched=True)}
    
    return lm_datasets

# ==============================================================================
# Data Collators
# ==============================================================================

@dataclass
class DataCollatorForMaskedDiffusion:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = None
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        input_ids = batch["input_ids"]
        
        B, L = input_ids.shape
        t = torch.rand(B, 1, device=input_ids.device)
        t = torch.clamp(t, 1e-4, 1.0)
        
        mask_prob = t.expand(B, L)
        mask_indices = torch.bernoulli(mask_prob).bool()
        
        labels = input_ids.clone()
        input_ids = input_ids.clone()
        input_ids[mask_indices] = self.tokenizer.mask_token_id
        
        labels[~mask_indices] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": batch["attention_mask"],
            "t": t 
        }

@dataclass
class DataCollatorForUniformDiffusion:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = None
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        input_ids = batch["input_ids"]
        B, L = input_ids.shape
        
        t = torch.rand(B, 1, device=input_ids.device)
        t = torch.clamp(t, 1e-4, 1.0)
        
        mask_prob = t.expand(B, L)
        corrupt_indices = torch.bernoulli(mask_prob).bool()
        
        # UDM Logic: Random Token Replacement
        random_noise = torch.randint(0, self.tokenizer.vocab_size, (B, L), device=input_ids.device)
        corrupted_ids = torch.where(corrupt_indices, random_noise, input_ids)
        
        labels = input_ids.clone()
        labels[~corrupt_indices] = -100
        
        return {
            "input_ids": corrupted_ids,
            "labels": labels,
            "attention_mask": batch["attention_mask"],
            "t": t
        }

@dataclass
class DataCollatorForBlockDiffusion:
    tokenizer: PreTrainedTokenizerBase
    block_size: int = 32
    pad_to_multiple_of: int = None
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        input_ids = batch["input_ids"] # x0
        B, L = input_ids.shape
        
        num_blocks = (L + self.block_size - 1) // self.block_size
        
        # Sample t per block
        t_blocks = torch.rand(B, num_blocks, device=input_ids.device)
        t_blocks = torch.clamp(t_blocks, 1e-4, 1.0)
        
        t_expanded = t_blocks.repeat_interleave(self.block_size, dim=1)[:, :L]
        
        mask_prob = t_expanded
        mask_indices = torch.bernoulli(mask_prob).bool()
        
        xt = input_ids.clone()
        xt[mask_indices] = self.tokenizer.mask_token_id
        
        # Dual-Stream Input: [Noisy, Clean]
        model_input = torch.cat([xt, input_ids], dim=1)
        
        labels = torch.full((B, 2*L), -100, dtype=input_ids.dtype, device=input_ids.device)
        labels_xt = input_ids.clone()
        labels_xt[~mask_indices] = -100
        labels[:, :L] = labels_xt
        
        return {
            "input_ids": model_input,
            "labels": labels,
            "t": t_expanded
        }

# ==============================================================================
# Trainer
# ==============================================================================

class DiffusionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        t = inputs.pop("t")
        
        # Clean up specialized inputs
        if "attention_mask" in inputs: inputs.pop("attention_mask")
        
        # Prepare timesteps
        t_mean = t.mean(dim=1) if t.ndim > 1 else t.squeeze()
        inputs["timesteps"] = t_mean
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Align BDM outputs if necessary
        if logits.shape[1] != labels.shape[1]:
            labels = labels[:, :logits.shape[1]]
            t = t[:, :logits.shape[1]]

        B, L, V = logits.shape
        
        per_tok_loss = F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
            reduction="none",
            ignore_index=-100
        ).view(B, L)
        
        loss = (per_tok_loss / (t + 1e-4)).sum() / ((labels != -100).sum() + 1e-6)
        
        if return_outputs:
            return loss, outputs
        return loss

class CSVLoggerCallback(transformers.TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.header_written = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Filter logs to save only loss and learning rate
            row = {k: v for k, v in logs.items() if k in ["loss", "learning_rate", "epoch", "step"]}
            if "epoch" not in row:
                row["epoch"] = state.epoch
            if "step" not in row:
                row["step"] = state.global_step
            
            if row:
                file_exists = os.path.isfile(self.log_path)
                with open(self.log_path, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    if not file_exists and not self.header_written:
                        writer.writeheader()
                        self.header_written = True
                    writer.writerow(row)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(model_args.config, trust_remote_code=True)
    
    # Model Selection
    if "ar" in model_args.config.lower():
        model = transformers.AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModel.from_config(config, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None: tokenizer.add_special_tokens({'mask_token': '<|mask|>'})

    lm_datasets = get_streaming_dataset(tokenizer, data_args, training_args)
    train_dataset = lm_datasets["train"]
    valid_dataset = lm_datasets["validation"]

    # Collator Selection
    if "udm" in model_args.config.lower():
        print("*** UDM: Random Token Corruption ***")
        data_collator = DataCollatorForUniformDiffusion(tokenizer=tokenizer)
        TrainerClass = DiffusionTrainer
    elif "bdm" in model_args.config.lower():
        print("*** BDM: Block Masking + Dual Stream Input ***")
        data_collator = DataCollatorForBlockDiffusion(tokenizer=tokenizer, block_size=32)
        TrainerClass = DiffusionTrainer
    elif "mdm" in model_args.config.lower():
        print("*** MDM: Standard Masking ***")
        data_collator = DataCollatorForMaskedDiffusion(tokenizer=tokenizer)
        TrainerClass = DiffusionTrainer
    else:
        # AR Model - Standard HF Trainer
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
        TrainerClass = Trainer

    # Determine log path based on config name
    config_name = os.path.splitext(os.path.basename(model_args.config))[0]
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"training_logs_{config_name}.csv")
    print(f"*** Saving training logs to {log_path} ***")

    trainer = TrainerClass(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        callbacks=[CSVLoggerCallback(log_path=log_path)]
    )
    
    model.config.use_cache = False

    if training_args.do_train:
        logging.info("*** Start Training ***")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.do_eval:
        logging.info("*** Evaluate on valid set***")
        metrics = trainer.evaluate(eval_dataset=valid_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        

if __name__ == "__main__":
    train()