from torch.utils.data import Dataset
import torch


class EmotionDataset(Dataset):
    def __init__(self, paths, tokz, label_vocab, logger, max_lengths=2048):
        self.logger = logger
        self.label_vocab = label_vocab
        self.max_lengths = max_lengths
        self.data = EmotionDataset.make_dataset(paths, tokz, label_vocab, logger, max_lengths)

    @staticmethod
    def make_dataset(paths, tokz, label_vocab, logger, max_lengths):
        logger.info('reading data from {}'.format(paths))
        dataset = []
        for path in paths:
            with open(path, 'r', encoding='utf8') as f:
                lines = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
                lines = [i.split('\t') for i in lines]
                for label, utt in lines:
                    # style, post, resp
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
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        res = dict()
        res['label'] = torch.LongTensor([i['label'] for i in batch])
        max_len = max([len(i['utt']) for i in batch])
        res['utt'] = torch.LongTensor([i['utt'] + [self.pad_id] * (max_len - len(i['utt'])) for i in batch])
        res['mask'] = torch.LongTensor([i['mask'] + [self.pad_id] * (max_len - len(i['mask'])) for i in batch])
        return res

if __name__ == '__main__':
    from transformers import BertTokenizer
    bert_path = '/home/data/tmp/bert-base-chinese'
    data_file = '/home/data/tmp/NLP_Course/TextBERT/data/test'
    label_vocab = '/home/data/tmp/NLP_Course/TextBERT/data/label_vocab'
    with open(label_vocab) as f:
        res = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
    label_vocab = dict(zip(res, range(len(res))))

    class Logger:
        def info(self, s):
            print(s)

    logger = Logger()
    tokz = BertTokenizer.from_pretrained(bert_path)
    dataset = EmotionDataset([data_file], tokz, label_vocab, logger)
    pad = PadBatchSeq(tokz.pad_token_id)
    print(pad([dataset[i] for i in range(5)]))
