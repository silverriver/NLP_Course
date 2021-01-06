from torch.utils.data import Dataset
import torch


class NLUDataset(Dataset):
    def __init__(self, paths, tokz, cls_vocab, slot_vocab, logger, max_lengths=2048):
        self.logger = logger
        self.data = NLUDataset.make_dataset(paths, tokz, cls_vocab, slot_vocab, logger, max_lengths)

    @staticmethod
    def make_dataset(paths, tokz, cls_vocab, slot_vocab, logger, max_lengths):
        logger.info('reading data from {}'.format(paths))
        dataset = []
        for path in paths:
            with open(path, 'r', encoding='utf8') as f:
                lines = [i.strip().lower() for i in f.readlines() if len(i.strip()) != 0]
                lines = [i.split('\t') for i in lines]
                for label, utt, slots in lines:
                    utt = tokz.convert_tokens_to_ids(list(utt)[:max_lengths])
                    slots = [slot_vocab[i] for i in slots.split()]
                    assert len(utt) == len(slots)
                    dataset.append([int(cls_vocab[label]),
                                    [tokz.cls_token_id] + utt + [tokz.sep_token_id], 
                                    tokz.create_token_type_ids_from_sequences(token_ids_0=utt),
                                    [tokz.pad_token_id] + slots + [tokz.pad_token_id]])
        logger.info('{} data record loaded'.format(len(dataset)))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        intent, utt, token_type, slot = self.data[idx]
        return {"intent": intent, "utt": utt, "token_type": token_type, "slot": slot}


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
        res['intent'] = torch.LongTensor([i['intent'] for i in batch])
        max_len = max([len(i['utt']) for i in batch])
        res['utt'] = torch.LongTensor([i['utt'] + [self.pad_id] * (max_len - len(i['utt'])) for i in batch])
        res['mask'] = torch.LongTensor([[1] * len(i['utt']) + [0] * (max_len - len(i['utt'])) for i in batch])
        res['token_type'] = torch.LongTensor([i['token_type'] + [self.pad_id] * (max_len - len(i['token_type'])) for i in batch])
        res['slot'] = torch.LongTensor([i['slot'] + [self.pad_id] * (max_len - len(i['slot'])) for i in batch])
        return PinnedBatch(res)


if __name__ == '__main__':
    from transformers import BertTokenizer
    bert_path = '/home/data/tmp/bert-base-chinese'
    data_file = '/home/data/tmp/NLP_Course/Joint_NLU/data/train.tsv'
    cls_vocab_file = '/home/data/tmp/NLP_Course/Joint_NLU/data/cls_vocab'
    slot_vocab_file = '/home/data/tmp/NLP_Course/Joint_NLU/data/slot_vocab'
    with open(cls_vocab_file) as f:
        res = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
    cls_vocab = dict(zip(res, range(len(res))))
    with open(slot_vocab_file) as f:
        res = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
    slot_vocab = dict(zip(res, range(len(res))))

    class Logger:
        def info(self, s):
            print(s)

    logger = Logger()
    tokz = BertTokenizer.from_pretrained(bert_path)
    dataset = NLUDataset([data_file], tokz, cls_vocab, slot_vocab, logger)
    pad = PadBatchSeq(tokz.pad_token_id)
    print(pad([dataset[i] for i in range(5)]))
