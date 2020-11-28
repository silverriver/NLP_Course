import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--bert_path', help='config file', default='/home/data/tmp/bert-base-chinese')
parser.add_argument('--save_path', help='training file', default='/home/data/tmp/NLP_Course/TextBERT/train')
parser.add_argument('--train_file', help='training file', default='/home/data/tmp/NLP_Course/TextBERT/data/train')
parser.add_argument('--valid_file', help='valid file', default='/home/data/tmp/NLP_Course/TextBERT/data/test')
parser.add_argument('--label_vocab', help='training file', default='/home/data/tmp/NLP_Course/TextBERT/data/label_vocab')

parser.add_argument('--lr', type=float, default=8e-6)
parser.add_argument('--lr_warmup', type=float, default=200)
parser.add_argument('--bs', type=int, default=70)
parser.add_argument('--batch_split', type=int, default=3)
parser.add_argument('--eval_steps', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--max_length', type=int, default=90)
parser.add_argument('--n_jobs', type=int, default=1, help='num of workers to process data')

parser.add_argument('--gpu', help='which gpu to use', type=str, default='3')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda', 0)

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import AdamW
import dataset
import utils
import traceback
from cls_trainer import ClsTrainer

train_path = os.path.join(args.save_path, 'train')
log_path = os.path.join(args.save_path, 'log')

def save_func(epoch, device):
    filename = utils.get_ckpt_filename('model', epoch)
    torch.save(trainer.state_dict(), os.path.join(train_path, filename))

try:
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    logger = utils.get_logger(os.path.join(args.save_path, 'train.log'))
    for path in [train_path, log_path]:
        if not os.path.isdir(path):
            logger.info('cannot find {}, mkdiring'.format(path))
            os.makedirs(path)

    for i in vars(args):
        logger.info('{}: {}'.format(i, getattr(args, i)))

    tokz = BertTokenizer.from_pretrained(args.bert_path)

    _, label2index, _ = utils.load_vocab(args.label_vocab)
    train_dataset = dataset.EmotionDataset([args.train_file], tokz, label2index, logger, max_lengths=args.max_length)
    valid_dataset = dataset.EmotionDataset([args.valid_file], tokz, label2index, logger, max_lengths=args.max_length)

    logger.info('Building models')
    bert_config = BertConfig.from_pretrained(args.bert_path)
    bert_config.num_labels = 8
    model = BertForSequenceClassification.from_pretrained(args.bert_path, config=bert_config)

    trainer = ClsTrainer(args, model, tokz, train_dataset, valid_dataset, log_path, logger, device)

    start_epoch = 0
    trainer.train(start_epoch, args.n_epochs, after_epoch_funcs=[save_func])
except:
    logger.error(traceback.format_exc())
