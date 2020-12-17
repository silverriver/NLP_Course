#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
        logger.info('reading data from {}'.format(paths))
        dataset = []
        for path in paths:
            with open(path, 'r', encoding='utf8') as f:
                lines = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
                lines = [i.split('\t') for i in lines]
                for line in lines:
                    # style, post, resp
                    dataset.append([int(line[0]),
                                    vocab.string2ids(' '.join(line[1].replace(' ', '')))[:max_lengths],
                                    vocab.string2ids(' '.join(line[2].replace(' ', '')))[:max_lengths]])
        logger.info('{} data record loaded'.format(len(dataset)))
        return dataset

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
