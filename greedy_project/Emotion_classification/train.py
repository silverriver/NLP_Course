# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2020-09-20
@LastEditTime: 2020-09-20
@LastEditors: Please set LastEditors
@Description: This file is for the NLP capstone of GreedyAI.com.
    This file implement the training interface of the emotion classifier.
@All Right Reserve
'''

import argparse
import torch
import os

parser = argparse.ArgumentParser()
# Setup the parameter parser
parser.add_argument('--bert_path', help='config file', default='/home/data/tmp/bert-base-chinese')
parser.add_argument('--save_path', help='training file', default='/home/data/tmp/NLP_Course/Emotion_classification/train')
parser.add_argument('--train_file', help='training file', default='/home/data/tmp/NLP_Course/Emotion_classification/data/train')
parser.add_argument('--valid_file', help='valid file', default='/home/data/tmp/NLP_Course/Emotion_classification/data/test')
parser.add_argument('--label_vocab', help='training file', default='/home/data/tmp/NLP_Course/TextBERT_multi_GPU/data/label_vocab')

parser.add_argument("--local_rank", help='used for distributed training', type=int, default=-1)
parser.add_argument('--lr', type=float, default=8e-6)
parser.add_argument('--lr_warmup', type=float, default=200)
parser.add_argument('--bs', type=int, default=70)
parser.add_argument('--batch_split', type=int, default=1)
parser.add_argument('--eval_steps', type=int, default=40)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--max_length', type=int, default=90)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--n_jobs', type=int, default=1, help='num of workers to process data')

parser.add_argument('--gpu', help='which gpu to use', type=str, default='0')

args = parser.parse_args()

# Setup for the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from transformers import BertConfig, BertTokenizer
from model import EmotionCLS
import dataset
import utils
import traceback
from cls_trainer import ClsTrainer
from torch.nn.parallel import DistributedDataParallel


if __name__ == '__main__':
    # Get the dir to save models and logs
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

            # Log out training settings
            for i in vars(args):
                logger.info('{}: {}'.format(i, getattr(args, i)))

        # Setup for distributed training
        distributed = (args.local_rank != -1)
        if distributed:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            torch.manual_seed(args.seed)
        else:
            device = torch.device("cuda", 0)
        
        # Init the tokenizer
        tokz = BertTokenizer.from_pretrained(args.bert_path)
        _, label2index, _ = utils.load_vocab(args.label_vocab)
        # Init the dataset
        train_dataset = dataset.EmotionDataset([args.train_file], tokz, label2index, logger, max_lengths=args.max_length)
        valid_dataset = dataset.EmotionDataset([args.valid_file], tokz, label2index, logger, max_lengths=args.max_length)

        # Init the model
        logger.info('Building models, rank {}'.format(args.local_rank))
        bert_config = BertConfig.from_pretrained(args.bert_path)
        bert_config.num_labels = 8
        model = EmotionCLS.from_pretrained(args.bert_path, config=bert_config).to(device)

        # Setup for distrbuted training
        if distributed:
            model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

        # Init the trainer
        trainer = ClsTrainer(args, model, tokz, train_dataset, valid_dataset, log_path, logger, device, distributed=distributed)

        # Start the training
        start_epoch = 0
        if args.local_rank in [-1, 0]:
            trainer.train(start_epoch, args.n_epochs, after_epoch_funcs=[save_func])
        else:
            trainer.train(start_epoch, args.n_epochs)

    except:
        logger.error(traceback.format_exc())
