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
from local_transformers_bioClinical_BERT.src.transformers.trainer import Trainer
from local_transformers_bioClinical_BERT.src.transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from local_transformers_bioClinical_BERT.src.transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from typing import Union, Any


# this CustomTrainer helps us overide the compute_loss of the original Trainer class. 
class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )

            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            additional_loss=outputs["additional_loss"] if isinstance(outputs, dict) else outputs[1]
            main_loss=outputs["main_loss"] if isinstance(outputs, dict) else outputs[2]

        
        return (loss, main_loss, additional_loss,  outputs) if return_outputs else (loss,main_loss, additional_loss)