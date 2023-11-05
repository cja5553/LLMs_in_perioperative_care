import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import BertForMaskedLM
from local_transformers_bioClinical_BERT.src.transformers.modeling_outputs import MaskedLMOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# this was added because was added because the original bioGPT will only input unsupervised features (without labels), any additional inputs will simply be filtered out. Hence, this is needed to ensure that the labels can now be accepted/ 

class CustomBioClinicalBertForCombinedLearning(BertForMaskedLM):
    def __init__(self, config, num_tasks=6, lambda_constant=10, weights=None):
        super().__init__(config)
        self.lambda_constant = lambda_constant
        self.weights = weights if weights is not None else [None]*num_tasks
        self.auxiliary = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(num_tasks)])

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        additional_labels=None,
        task_ids=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        auxs = []
        additional_losses = []
        # print(task_ids)
        if task_ids is not None:
            for task_id in range(len(self.auxiliary)):
                task_specific_layer = self.auxiliary[task_id]
                task_specific_indices = torch.where(task_ids == task_id)[0]
                task_specific_sequence_output = sequence_output[task_specific_indices]
                aux_output = task_specific_layer(task_specific_sequence_output)
                aux_output_pooled = torch.mean(aux_output, dim=1)
                auxs.append(aux_output_pooled)
                
                if additional_labels is not None:
                    task_specific_additional_labels = additional_labels[task_specific_indices]
                    if self.weights[task_id] is not None:
                        additional_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.weights[task_id])
                        additional_loss = additional_loss_fct(aux_output_pooled.view(-1), task_specific_additional_labels.view(-1).float())
                        additional_loss = additional_loss
                    else:
                        additional_loss_fct = torch.nn.BCEWithLogitsLoss()
                        # print(aux_output_pooled)
                        additional_loss = additional_loss_fct(aux_output_pooled.view(-1), task_specific_additional_labels.view(-1).float())
                    additional_losses.append(additional_loss)

            additional_loss = torch.stack(additional_losses).mean() if additional_losses else 0
        else:
            additional_loss = 0
        total_loss,masked_lm_loss = None, None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            additional_loss=self.lambda_constant * additional_loss
            # print("masked lm loss:", masked_lm_loss)
            # print("additional_loss", additional_loss)
            # w
            total_loss = masked_lm_loss + additional_loss
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((total_loss, additional_loss, masked_lm_loss) + output)

        return MaskedLMOutput(
            loss=total_loss,
            additional_loss=additional_loss,
            main_loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
