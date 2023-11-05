from transformers import BioGptForCausalLM
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import BertForMaskedLM
from local_transformers_clinical_BERT.src.transformers.modeling_outputs import MaskedLMOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# this was added because was added because the original bioGPT will only input unsupervised features (without labels), any additional inputs will simply be filtered out. Hence, this is needed to ensure that the labels can now be accepted/ 

class CustomClinicalBertForCombinedLearning(BertForMaskedLM):
    def __init__(self, *args, lambda_constant=1.0, weights=None, **kwargs):
        return_dict = kwargs.pop("return_dict", None)
        super().__init__(*args, **kwargs)
        self.auxiliary = nn.Linear(self.config.hidden_size, 1)
        self.lambda_constant = lambda_constant
        self.weights = weights
        self.config.return_dict = return_dict if return_dict is not None else self.config.return_dict

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        additional_labels:Optional[torch.Tensor]=None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

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
        aux_logits = self.auxiliary(sequence_output)
        # print(aux_logits.shape)
        avg_aux_logits = torch.mean(aux_logits, dim=1)
        prediction_scores = self.cls(sequence_output)
        # print("Shape of aux_logits: ", aux_logits.shape)
        # print("Shape of additional_labels: ", additional_labels.shape)

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            if self.weights is not None:
                # print(additional_labels.view(-1))
                additional_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.weights)
            else:
                additional_loss_fct = torch.nn.BCEWithLogitsLoss()
            additional_loss = additional_loss_fct(avg_aux_logits.view(-1), additional_labels.view(-1))
            additional_loss=self.lambda_constant*additional_loss
            main_loss=masked_lm_loss
            masked_lm_loss=masked_lm_loss+(additional_loss)
            # print("main_loss",main_loss)
            # print("additional_loss",additional_loss)
            # w
        else:
            masked_lm_loss = None
            additional_loss = None
            main_loss = None
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,additional_loss,main_loss) + output) 

        return MaskedLMOutput(
            loss=masked_lm_loss,
            additional_loss=additional_loss,
            main_loss=main_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
