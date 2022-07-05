# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2022-05-27
@LastEditTime: 2022-05-27
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the function to calculate label smoothing loss.
@All Right Reserve
'''

import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_labels, smoothing=0.0, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        assert 0 <= smoothing <= 1

        self.ignore_index = ignore_index
        self.confidence = 1 - smoothing

        if smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction='batchmean')
            n_ignore_idxs = 1 + (ignore_index >= 0)   # 1 for golden truth, later one for ignore_index
            one_hot = torch.full((1, n_labels), fill_value=(smoothing / (n_labels - n_ignore_idxs)))
            if ignore_index >= 0:
                one_hot[0, ignore_index] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            self.criterion = nn.NLLLoss(reduction='mean', ignore_index=ignore_index)
        
    def forward(self, log_inputs, targets):
        '''
        Task2
        Calculate the Label Smoothing Loss.
        Take a small amount of confidence from the golden labels (the value of smoothing) and uniformly
        distribution this confidence to the other labels.

        This function should be implemented as follow:
        1. if confidence < 1, then calculate the target distribution by taking a small amount of confidence 
           from the golden labels (the value of smoothing) and uniformly distribution this confidence to the other labels.
           for example, if confidence = 0.1, n_labels = 5, the golden label is 2. 
           Then the smoothed target distribution should be [0.025, 0.025, 0.9, 0.025, 0.025]
           Note: you can use tensor.scatter_ here.
        2. mask the confidence of the ignore_index to 0.
        3. calculate the loss by using the KLDivLoss
        '''
        if self.confidence < 1:
            tdata = targets.data
  
            tmp = self.one_hot.repeat(targets.shape[0], 1)
            tmp.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if self.ignore_index >= 0:
                mask = torch.nonzero(tdata.eq(self.ignore_index)).squeeze(-1)
                if mask.numel() > 0:
                    tmp.index_fill_(0, mask, 0)

            targets = tmp
        
        return self.criterion(log_inputs, targets)
