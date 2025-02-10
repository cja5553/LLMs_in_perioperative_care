from transformers import BioGptForCausalLM
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from local_transformers.src.transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BioGptModel
import statistics
# this was added because was added because the original bioGPT will only input unsupervised features (without labels), any additional inputs will simply be filtered out. Hence, this is needed to ensure that the labels can now be accepted/ 



class CustomBioGptForCausalLM(BioGptForCausalLM):
    _keys_to_ignore_on_load_missing = ["output_projection.weight"]
    _tied_weights_keys = ["output_projection.weight"]

    def __init__(self,config, lambda_constant=10,num_tasks=6,weights=None):
        super().__init__(config)
        self.biogpt = BioGptModel(config)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.weights = weights if weights is not None else [None]*num_tasks
        self.lambda_constant = lambda_constant
        self.auxiliary = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(num_tasks)])

    def get_output_embeddings(self):
        return self.output_projection

    def set_output_embeddings(self, new_embeddings):
        self.output_projection = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        additional_labels:Optional[torch.LongTensor] = None,
        task_ids: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.biogpt(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_output = outputs[0]
        prediction_scores = self.output_projection(sequence_output)
        auxs = []
        additional_losses = []
        # print(task_ids)
        # print(labels)
        # print(additional_labels)
        # # w
        if task_ids is not None:
            for task_id in range(len(self.auxiliary)):
                # print(task_id)
                task_specific_layer = self.auxiliary[task_id]
                task_specific_indices = torch.where(task_ids == task_id)[0]
                # print(task_specific_indices)
                task_specific_outputs = sequence_output[task_specific_indices]
                aux_output = task_specific_layer(task_specific_outputs)
                aux_output_pooled = torch.mean(aux_output, dim=1)
                auxs.append(aux_output_pooled)
                if additional_labels is not None:
                    task_specific_additional_labels = additional_labels[task_specific_indices]
                    if self.weights[task_id] is not None:
                        additional_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.weights[task_id])
                        additional_loss = additional_loss_fct(aux_output_pooled.view(-1), task_specific_additional_labels.view(-1).float())
                    else:
                        additional_loss_fct = torch.nn.BCEWithLogitsLoss()
                        additional_loss = additional_loss_fct(aux_output_pooled.view(-1), task_specific_additional_labels.view(-1).float())
                    additional_losses.append(additional_loss)

            additional_loss = torch.stack(additional_losses).mean() if additional_losses else 0
        else:
            additional_loss = 0

        total_loss,lm_loss,main_loss = None, None, None
        if labels is not None:
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            main_loss=lm_loss
            additional_loss=additional_loss*self.lambda_constant
            total_loss = main_loss + additional_loss

        # print("lm_loss",lm_loss)
        # print("additional_loss",additional_loss)
        # w
        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((total_loss,additional_loss,main_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            total_loss=total_loss,
            main_loss=main_loss,
            additional_loss=additional_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

