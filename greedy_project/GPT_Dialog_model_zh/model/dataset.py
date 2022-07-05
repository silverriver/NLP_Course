# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2022-05-27
@LastEditTime: 2022-05-27
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the dataset class that is used to read and parse the data.
@All Right Reserve
'''

from torch.utils.data import Dataset
import torch


class DialogDataset(Dataset):
    def __init__(self, paths, vocab, logger, max_lengths=2048):
        self.logger = logger
        self.vocab = vocab
        self.max_lengths = max_lengths
        self.data = DialogDataset.make_dataset(paths, vocab, logger, max_lengths)

    @staticmethod
    def make_dataset(paths, vocab, logger, max_lengths):
        '''
        Task1
        Load the dataset into memory.
        inputs:
            paths: a list of paths to the data files.
            vocab: the tokenizer object that has a string2ids method to convert string to ids.
            logger: the logger object that has a log method.
            max_lengths: the maximum length of the post and response.
        output:
            data: a list of tuples (style, post, resp).
            Note: style is here for a legacy reasone, it is not used in other part of the project.
            post and resp are lists of token ids.
        
        This function should be implemented as follow:
        1. iterate through the paths list
        2. for each path, load the data into memory
        3. for each line of the data, split the line using '\t' and convert the second and third column to ids using vocab.string2ids
        4. return all the tokenized data as a list of tuples (style, post, resp) (you can set the style to be any value you like)
        '''
        ###############################
        # YOUR CODE HERE for Task 1   #
        ###############################
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        style, post, resp = self.data[idx]
        post = [self.vocab.eos_id] + post + [self.vocab.eos_id]
        resp = [self.vocab.eos_id] + resp + [self.vocab.eos_id]
        return {"style": style, "post": post, "post_len": len(post), "resp": resp, "resp_len": len(resp)}


class PadBatchSeq:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        res = dict()
        res['style'] = torch.LongTensor([i['style'] for i in batch])
        res['post_len'] = torch.LongTensor([i['post_len'] for i in batch])
        res['resp_len'] = torch.LongTensor([i['resp_len'] for i in batch])
        post_max_len = max([len(i['post']) for i in batch])
        resp_max_len = max([len(i['resp']) for i in batch])
        res['post'] = torch.LongTensor([i['post'] + [self.pad_id] * (post_max_len - len(i['post'])) for i in batch])
        res['resp'] = torch.LongTensor([i['resp'] + [self.pad_id] * (resp_max_len - len(i['resp'])) for i in batch])
        return res
