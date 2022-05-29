# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2022-05-27
@LastEditTime: 2022-05-27
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the Slot Tagging model.
@All Right Reserve
'''

from transformers import BertPreTrainedModel, BertModel
from torch import nn


class BERTSlotModule(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_slot_labels = config.num_slot_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.slot_classifier = nn.Linear(config.hidden_size, config.num_slot_labels)

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

        seq_encoding = outputs[0]
        slot_logits = self.slot_classifier(seq_encoding)

        return slot_logits


class RNNSlotModule(nn.Module):
    def __init__(self, hidden_size, num_layers, num_slot_labels, vocab_size, dropout=0.1):
        super().__init__()
        self.num_slot_labels = num_slot_labels
        self.embed = nn.Embedding(vocab_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.slot_classifier = nn.Linear(hidden_size, num_slot_labels)
        self.lstm = nn.LSTM(hidden_size, hidden_size//2, num_layers, batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, input_ids, **kwargs):
        '''
        :param input_ids (batch_size, sequence_length)
        '''
        x_embed = self.embed(input_ids)
        x_embed = self.dropout(x_embed)
        x_lstm, _ = self.lstm(x_embed)

        slot_logits = self.slot_classifier(x_lstm)

        return slot_logits