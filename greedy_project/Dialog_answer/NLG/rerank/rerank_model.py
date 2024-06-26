# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2022-05-27
@LastEditTime: 2022-05-27
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the Rerank model.
@All Right Reserve
'''

from transformers import BertPreTrainedModel, BertModel
from torch import nn
import torch.nn.functional as F
import torch


class RerankModule(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.intent_classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        """
        :param input_ids: (batch_size, sequence_length)
        :param attention_mask: (batch_size, sequence_length)
        :param token_type_ids: (batch_size, sequence_length)
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = outputs[1]
        rank_logits = self.intent_classifier(pooled_output)
        return rank_logits
