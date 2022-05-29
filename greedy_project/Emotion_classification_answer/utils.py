# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2020-09-20
@LastEditTime: 2020-09-20
@LastEditors: Please set LastEditors
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the utilities used for the emotion classifier.
@All Right Reserve
'''

import os
import json
import random
import torch
import logging
import argparse
from torch.utils.checkpoint import checkpoint
from attrdict import AttrDict


def get_logger(filename, print2screen=True):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] \
>> %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    if print2screen:
        logger.addHandler(ch)
    return logger


def load_vocab(vocab_file):
    with open(vocab_file) as f:
        res = [i.strip().lower() for i in f.readlines() if len(i.strip()) != 0]
    return res, dict(zip(res, range(len(res)))), dict(zip(range(len(res)), res))  # list, token2index, index2token


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def load_config(config_file):
    with open(config_file) as f:
        config = json.load(f)
        return AttrDict(config)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def get_latest_ckpt(dir_name):
    files = [i for i in os.listdir(dir_name) if '.ckpt' in i]
    if len(files) == 0:
        return None
    else:
        res = ''
        num = -1
        for i in files:
            try:
                n = int(i.split('-')[-1].split('.')[0])
                if n > num:
                    num = n
                    res = i
            except ValueError:
                pass
        return res


def get_epoch_from_ckpt(ckpt):
    return int(ckpt.split('-')[-1].split('.')[0])


def get_ckpt_filename(name, epoch):
    return '{}-{}.ckpt'.format(name, epoch)


def get_ckpt_step_filename(name, step):
    return '{}-{}-step.ckpt'.format(name, step)

