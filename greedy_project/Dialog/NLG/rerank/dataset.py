# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2022-05-27
@LastEditTime: 2022-05-27
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the Rerank model.
@All Right Reserve
'''

from torch.utils.data import Dataset
import torch


class RerankDataset(Dataset):
    def __init__(self, paths, tokz, logger, max_lengths=2048):
        self.logger = logger
        self.data = RerankDataset.make_dataset(paths, tokz, logger, max_lengths)

    @staticmethod
    def make_dataset(paths, tokz, logger, max_lengths):
        logger.info('reading data from {}'.format(paths))
        dataset = []
        for path in paths:
            with open(path, 'r', encoding='utf8') as f:
                lines = [i.strip().split('\t') for i in f.readlines() if len(i.strip()) != 0]
                lines = [i for i in lines if len(i) == 3]
                
                for label, query, response in lines:
                    if label == '1':
                        assert not dataset or len(dataset[-1][2]) != 0
                        dataset.append([query, [response], []])
                    else:
                        dataset[-1][-1].append(response)
        _dataset = []
        for query, pos_list, neg_list in dataset:
            query = tokz.convert_tokens_to_ids(list(query.replace(' ', ''))[:max_lengths])
            pos_list = [tokz.convert_tokens_to_ids(list(post.replace(' ', ''))[:max_lengths]) for post in pos_list]
            neg_list = [tokz.convert_tokens_to_ids(list(neg.replace(' ', ''))[:max_lengths]) for neg in neg_list]
            res = {'pos_list': [], 'neg_list': []}
            for pos in pos_list:
                res['pos_list'].append(
                    [tokz.build_inputs_with_special_tokens(token_ids_0=query, token_ids_1=pos), 
                     tokz.create_token_type_ids_from_sequences(token_ids_0=query, token_ids_1=pos)])
            for neg in neg_list:
                res['neg_list'].append(
                    [tokz.build_inputs_with_special_tokens(token_ids_0=query, token_ids_1=neg), 
                     tokz.create_token_type_ids_from_sequences(token_ids_0=query, token_ids_1=neg)])
            _dataset.append(res)
        logger.info('{} data record loaded'.format(len(_dataset)))
        return _dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # {'pos_list': [], 'neg_list': []}


class PinnedBatch:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, k):
        return self.data[k]

    def pin_memory(self):
        for k in self.data.keys():
            self.data[k] = self.data[k].pin_memory()
        return self
    
    def __str__(self):
        return str(self.data)

    def __repr__(self) -> str:
        return str(self.data)


class PadBatchSeq:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        res = dict()
        res['utt'] = []
        res['token_type'] = []
        res['label'] = []
        for i in batch:
            for j in range(len(i['pos_list'])):
                res['utt'].append(i['pos_list'][j][0])
                res['token_type'].append(i['pos_list'][j][1])
                res['label'].append(1)
            for j in range(len(i['neg_list'])):
                res['utt'].append(i['neg_list'][j][0])
                res['token_type'].append(i['neg_list'][j][1])
                res['label'].append(0)

        max_len = max([len(i) for i in res['utt']])
        res['mask'] = torch.LongTensor([[1] * len(i) + [0] * (max_len - len(i)) for i in res['utt']])
        res['utt'] = torch.LongTensor([i + [self.pad_id] * (max_len - len(i)) for i in res['utt']])
        res['token_type'] = torch.LongTensor([i + [self.pad_id] * (max_len - len(i)) for i in res['token_type']])
        res['label'] = torch.FloatTensor(res['label'])
        return PinnedBatch(res)


if __name__ == '__main__':
    from transformers import BertTokenizer
    bert_path = 'bert-base-chinese'
    data_file = 'data/E-commerce-dataset/dev.txt'

    class Logger:
        def info(self, s):
            print(s)

    logger = Logger()
    tokz = BertTokenizer.from_pretrained(bert_path)
    dataset = RerankDataset([data_file], tokz, logger)
    pad = PadBatchSeq(tokz.pad_token_id)
    print(pad([dataset[i] for i in range(5)]))
