from typing import List, Dict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import default_data_collator
import pandas as pd
import torch
import random
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
import itertools
from torch.utils.data import TensorDataset
import tensorflow as tf
from transformers import TrainingArguments, BioGptTokenizer, BioGptForCausalLM
import torch.nn as nn
from local_transformers.src.transformers.trainer import Trainer
from typing import Union, Any


# because we want to take in the labels, we have to define a custom_data_collator that will allow us to accept this inpute
def custom_data_collator(features: List[Dict], tokenizer: PreTrainedTokenizerBase):
    batch = {}
    input_ids = [feature["input_ids"] for feature in features]
    attention_mask = [feature["attention_mask"] for feature in features]
    labels = [feature["labels"] for feature in features]
    additional_labels = [feature["additional_labels"] for feature in features]

    batch["input_ids"] = tokenizer.pad(input_ids, return_tensors="pt")["input_ids"]
    batch["attention_mask"] = tokenizer.pad(attention_mask, return_tensors="pt")["input_ids"]
    batch["labels"] = tokenizer.pad(labels, return_tensors="pt")["input_ids"].squeeze(-1) # remove the extra dimension
    batch["additional_labels"] = torch.tensor(additional_labels, dtype=torch.float)

    return batch

# this CustomTrainer helps us overide the compute_loss of the original Trainer class. 
class CustomTrainer(Trainer):
    def __init__(self, *args, tokenizer, lambda_constant=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.lambda_constant = lambda_constant
        self.num_features = self.model.config.hidden_size

    def collate_fn(self, features):
        return custom_data_collator(features, self.tokenizer)
    
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # note: the next few lines was adapted from https://huggingface.co/docs/transformers/main_classes/trainer
        labels = inputs.pop("labels")

        # obtain the additional labels from the inputs
        additional_labels=inputs.pop("additional_labels").float() 
        task_ids=inputs.pop("task_ids")

        outputs = model(**inputs, labels=labels, additional_labels=additional_labels,task_ids=task_ids)

        total_loss = outputs.total_loss
        additional_loss=outputs.additional_loss
        main_loss=outputs.main_loss
        # print(additional_loss)
        # w
        return (main_loss, additional_loss, total_loss, outputs) if return_outputs else (main_loss, additional_loss, total_loss)