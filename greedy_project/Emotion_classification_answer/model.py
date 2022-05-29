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

        # Encoder the input sequence with BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Get the representation for input utterances
        pooled_output = outputs[1]  # (batch_size, hidden_size)

        pooled_output = self.dropout(pooled_output)
        # Get the logits for classification
        logits = self.classifier(pooled_output)   # (batch_size, num_labels)

        if labels is not None:
            # If labels are available, then compute the loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            # If labels are not available, then set loss to None
            loss = None

        return loss, logits
