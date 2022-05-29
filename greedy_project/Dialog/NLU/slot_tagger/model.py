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
        ########################################################
        # 4. TODO complete the code for a BERT slot tagger     #
        # follow the following steps:                          #  
        # 1. call a forward pass of bert                       #
        # 2. get the hidden state for each token               #
        # 3. apply the output layer                            #
        # 4. return the logits                                 #
        ########################################################
        raise NotImplementedError


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
        ########################################################
        # 3. TODO complete the code for a LSTM slot tagger     #
        # follow the following steps:                          #  
        # 1. embed the input_ids                               #
        # 2. apply the dropout layer                           #
        # 3. apply the LSTM layer                              #
        # 4. apply the output layer                            #
        # 5. return the logits                                 #
        ########################################################
        raise NotImplementedError
