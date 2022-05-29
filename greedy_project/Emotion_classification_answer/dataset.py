# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2020-09-20
@LastEditTime: 2020-09-20
@LastEditors: Please set LastEditors
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the dataset used of the emotion classifier.
@All Right Reserve
'''

from torch.utils.data import Dataset
import torch


class EmotionDataset(Dataset):
    def __init__(self, paths, tokz, label_vocab, logger, max_lengths=2048):
        # Init attributes
        self.logger = logger
        self.label_vocab = label_vocab
        self.max_lengths = max_lengths
        # Read all the data into memory
        self.data = EmotionDataset.make_dataset(paths, tokz, label_vocab, logger, max_lengths)

    @staticmethod
    def make_dataset(paths, tokz, label_vocab, logger, max_lengths):
        logger.info('reading data from {}'.format(paths))
        dataset = []
        for path in paths:
            with open(path, 'r', encoding='utf8') as f:
                # Read each line of the input file and filter out the empty line
                lines = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
                lines = [i.split('\t') for i in lines]
                for label, utt in lines:
                    # Emotion, Utterance, Attention_Mask
                    utt = tokz(utt[:max_lengths])
                    dataset.append([int(label_vocab[label]),
                                    utt['input_ids'], 
                                    utt['attention_mask']])
        logger.info('{} data record loaded'.format(len(dataset)))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, utt, mask = self.data[idx]
        return {"label": label, "utt": utt, "mask": mask}


class PadBatchSeq:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        # Pad the batch with self.pad_id
        res = dict()
        res['label'] = torch.LongTensor([i['label'] for i in batch])
        max_len = max([len(i['utt']) for i in batch])
        res['utt'] = torch.LongTensor([i['utt'] + [self.pad_id] * (max_len - len(i['utt'])) for i in batch])
        res['mask'] = torch.LongTensor([i['mask'] + [self.pad_id] * (max_len - len(i['mask'])) for i in batch])
        return res
