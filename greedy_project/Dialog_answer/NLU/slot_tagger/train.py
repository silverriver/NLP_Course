# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2022-05-27
@LastEditTime: 2022-05-27
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the Slot Tagging model.
@All Right Reserve
'''

import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--bert_path', help='config file', default='bert-base-chinese')
parser.add_argument('--save_path', help='path to save checkpoints', default='rnn_slot')
parser.add_argument('--train_file', help='training data', default='data/NLU/train.txt')
parser.add_argument('--valid_file', help='valid data', default='data/NLU/dev.txt')
parser.add_argument('--slot_label_vocab', help='training file', default='data/NLU/slot_vocab.txt')
parser.add_argument('--base_model', help='which base_model to use (RNN/BERT)', default='RNN')

parser.add_argument("--local_rank", help='used for distributed training', type=int, default=-1)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--warmup_steps', type=float, default=100)
parser.add_argument('--bs', type=int, default=30)
parser.add_argument('--batch_split', type=int, default=1)
parser.add_argument('--eval_steps', type=int, default=40)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--max_length', type=int, default=90)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--n_jobs', type=int, default=1, help='num of workers to process data')

parser.add_argument('--gpu', help='which gpu to use', type=str, default='3')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from transformers import BertConfig, BertTokenizer
from model import BERTSlotModule, RNNSlotModule
import dataset
import utils
import traceback
from trainer import Trainer
from torch.nn.parallel import DistributedDataParallel


utils.seed_everything(args.seed)
train_path = os.path.join(args.save_path, 'train')
log_path = os.path.join(args.save_path, 'log')

def save_func(epoch, device):
    filename = utils.get_ckpt_filename('model', epoch)
    torch.save(trainer.state_dict(), os.path.join(train_path, filename))

try:
    if args.local_rank == -1 or args.local_rank == 0:
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
    while not os.path.isdir(args.save_path):
        pass
    logger = utils.get_logger(os.path.join(args.save_path, 'train.log'))

    if args.local_rank == -1 or args.local_rank == 0:
        for path in [train_path, log_path]:
            if not os.path.isdir(path):
                logger.info('cannot find {}, mkdiring'.format(path))
                os.makedirs(path)

        for i in vars(args):
            logger.info('{}: {}'.format(i, getattr(args, i)))

    distributed = (args.local_rank != -1)
    if distributed:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.manual_seed(args.seed)
    else:
        device = torch.device("cuda", 0)  # if you want to train on CPU, change to torch.device("cpu")
    tokz = BertTokenizer.from_pretrained(args.bert_path)
    _, slot2index, _ = utils.load_vocab(args.slot_label_vocab)
    train_dataset = dataset.NLUDataset([args.train_file], tokz, slot2index, logger, max_lengths=args.max_length)
    valid_dataset = dataset.NLUDataset([args.valid_file], tokz, slot2index, logger, max_lengths=args.max_length)

    logger.info('Building models, rank {}'.format(args.local_rank))

    if args.base_model == 'RNN':
        logger.info('Building {} models, rank {}'.format(args.base_model, args.local_rank))
        model = RNNSlotModule(768, 2, len(slot2index), len(tokz), dropout=0.1)
        model = model.to(device)
    elif args.base_model == 'BERT':
        logger.info('Building {} models, rank {}'.format(args.base_model, args.local_rank))
        bert_config = BertConfig.from_pretrained(args.bert_path)
        bert_config.num_slot_labels = len(slot2index)
        model = BERTSlotModule.from_pretrained(args.bert_path, config=bert_config).to(device)
    else:
        logger.info('Can only use RNN or BERT as base models')

    if distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    trainer = Trainer(args, model, tokz, train_dataset, valid_dataset, log_path, logger, device, distributed=distributed)

    start_epoch = 0
    if args.local_rank in [-1, 0]:
        trainer.train(start_epoch, args.n_epochs, after_epoch_funcs=[save_func])
    else:
        trainer.train(start_epoch, args.n_epochs)

except:
    logger.error(traceback.format_exc())
