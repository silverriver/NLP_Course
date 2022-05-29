# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2020-09-20
@LastEditTime: 2020-09-20
@LastEditors: Please set LastEditors
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the model of the emotion classifier.
@All Right Reserve
'''

from transformers import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import CrossEntropyLoss


class EmotionCLS(BertPreTrainedModel):
    def __init__(self, config):
        # Init the super class
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Init the bert encoder
        self.bert = BertModel(config)
        classifier_dropout = (config.hidden_dropout_prob)
        # Init the dropout layer
        self.dropout = nn.Dropout(classifier_dropout)
        # Init the classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Init other weights
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        """Define forward pass for the model.
        Args:
            input_ids: input token ids. `torch.LongTensor` of shape [batch_size, sequence_length] 
            attention_mask: mask for input ids. `torch.LongTensor` of shape [batch_size, sequence_length] 
            token_type_ids: token types for input token ids. `torch.LongTensor` of shape [batch_size, sequence_length]
            labels: labels for the input token ids. `torch.LongTensor` of shape [batch_size, sequence_length]
        Returns:
            loss: the loss value of the model. loss=None if labels==None
            logits: logits for the classification `torch.FloatTensor` of shape [batch_size, num_labels].
        """
        loss, logits = None, None

        #################################################
        # 3. DOTO - Add the code to do the forward pass #
        #################################################

        return loss, logits
