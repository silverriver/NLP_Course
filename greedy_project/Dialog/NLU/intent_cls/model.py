# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2022-05-27
@LastEditTime: 2022-05-27
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the Intent Classification model.
@All Right Reserve
'''

from transformers import BertPreTrainedModel, BertModel
from torch import nn
import torch.nn.functional as F
import torch


class BertIntentModule(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_intent_labels = config.num_intent_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, config.num_intent_labels)

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
        # 2. TODO complete the code for a BERT text classifier #
        # follow the following steps:                          #  
        # 1. call a forward pass of bert                       #
        # 2. pool the hidden state of [CLS]                    #
        # 3. apply the dropout layer                           #
        # 4. apply the output layer                            #
        # 5. return the logits                                 #
        ########################################################
        raise NotImplementedError


class Dense(nn.Module):
    def __init__(self, in_size, out_size, activation=F.relu):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.linear(x))


class CNNIntentModule(nn.Module):
    def __init__(self, feature_size, kernel_size, embedding_size, vocab_size, fc_size, num_cls, dropout_rate):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embedding_size)
        self.dropout = dropout_rate
        self.convs = nn.ModuleList([nn.Conv1d(embedding_size, fs, ks) for fs, ks in zip(feature_size, kernel_size)])
        self.dropout = nn.Dropout(dropout_rate)
        fc_size = list(fc_size)
        fc_size = list(zip([sum(feature_size)] + fc_size[1:], fc_size))
        self.fc = nn.ModuleList([Dense(i, j) for i, j in fc_size])
        self.output_layer = nn.Linear(fc_size[-1][-1], num_cls)


    def forward(self, input_ids, attention_mask, **kwargs):
        '''
        :param input_ids: (batch_size, sequence_length)
        :param attention_mask: (batch_size, sequence_length)
        '''
        #######################################################
        # 1. TODO complete the code for a CNN text classifier #
        # follow the following steps:                         #  
        # 1. embed the input_ids                              #
        # 2. apply the convolutional layers                   #
        # 3. apply the max pooling layer                      #
        # 4. apply the dropout layer                          #
        # 5. apply the fully connected layer                  #
        # 6. apply the output layer                           #
        # 7. return the logits                                #
        #######################################################
        raise NotImplementedError

