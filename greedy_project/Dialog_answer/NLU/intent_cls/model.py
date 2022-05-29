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

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        return intent_logits


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
        x_embed = self.embed(input_ids) * attention_mask.unsqueeze(2)
        x_embed = x_embed.permute([0, 2, 1])   # [bs, embed_size, len]
        x_embed = [conv(x_embed) for conv in self.convs]  # [(bs, fs, len), ...]
        x_embed = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_embed]  # [(bs, fs), ...]
        x_embed = torch.cat(x_embed, 1)
        x_embed = self.dropout(x_embed)
        for fc in self.fc:
            x_embed = fc(x_embed)
        x_embed = self.dropout(x_embed)
        logits = self.output_layer(x_embed)
        return logits   # [bs, logits]

