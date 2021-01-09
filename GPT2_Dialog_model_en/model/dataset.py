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
import json
import os


class DialogDataset(Dataset):
    def __init__(self, paths, tokz, logger, max_context_len=100, max_resp_len=100):
        self.logger = logger
        self.tokz = tokz
        self.data = self.make_dataset(paths, tokz, logger, max_context_len, max_resp_len)

    @staticmethod
    def make_dataset(paths, tokz, logger, max_context_len, max_resp_len):
        logger.info('reading data from {}'.format(paths))
        dataset = []
        skipped_line = 0
        for path in paths:
            lines = []
            with open(path, 'r', encoding='utf8') as f:
                for line in [json.loads(i) for i in f.readlines() if len(i.strip()) != 0]:
                    lines.append([tokz.convert_tokens_to_ids(tokz.tokenize(i)) for i in line])

            for line in lines:
                if len(line) <= 1:
                    skipped_line += 1
                    continue

                for index in range(1, len(line)):
                    context = []
                    context_seg_id = []
                    for i in range(index):
                        context += line[i] + [tokz.sep_token_id]
                        context_seg_id += ([tokz.sp1_token_id if i % 2 == 0 else tokz.sp2_token_id] * (len(line[i]) + 1))
                    context = [tokz.cls_token_id] + context[-max_context_len:]
                    context_seg_id = [tokz.sp1_token_id] + context_seg_id[-max_context_len:]

                    resp = [tokz.bos_token_id] + line[index][:max_resp_len] + [tokz.eos_token_id]
                    dataset.append([context, context_seg_id, resp])

        logger.info('{} data record loaded, {} lines skipped'.format(len(dataset), skipped_line))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, context_seg_id, resp = self.data[idx]
        return {"context": context, "context_seg_id": context_seg_id, "resp": resp}

class PinnedBatch:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, k):
        return self.data[k]

    def pin_memory(self):
        for k in self.data.keys():
            self.data[k] = self.data[k].pin_memory()
        return self

class PadBatchSeq:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        res = dict()
        context_max_len = max([len(i['context']) for i in batch])
        resp_max_len = max([len(i['resp']) for i in batch])
        res['context'] = torch.LongTensor([i['context'] + [self.pad_id] * (context_max_len - len(i['context'])) for i in batch])
        res['context_seg_id'] = torch.LongTensor([i['context_seg_id'] + [self.pad_id] * (context_max_len - len(i['context_seg_id'])) for i in batch])
        res['resp'] = torch.LongTensor([i['resp'] + [self.pad_id] * (resp_max_len - len(i['resp'])) for i in batch])
        return PinnedBatch(res)


if __name__ == '__main__':
    dialogpt_dir = '/home/someone/dialogpt_small'
    dialog_file = '/home/someone/data/dialog_valid.json'
    from text import Tokenizer
    tokz = Tokenizer.from_pretrained(dialogpt_dir)

    class Logger():
        def info(self, s):
            print(s)
    logger = Logger()

    dataset = DialogDataset([dialog_file], tokz, logger, 100, 100)

    import random
    for i in random.sample(range(len(dataset)), 3):
        res = dataset[i]
        print(len(res['context']), tokz.decode(res['context']))
        print(len(res['context_seg_id']), tokz.decode(res['context_seg_id']))
        print(len(res['resp']), tokz.decode(res['resp']))

    res = PadBatchSeq(tokz.pad_token_id)([dataset[i] for i in random.sample(range(len(dataset)), 5)])
    print(res)
