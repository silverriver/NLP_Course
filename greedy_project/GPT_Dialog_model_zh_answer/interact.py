# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2022-05-27
@LastEditTime: 2022-05-27
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement a script that can be used to interact with a trained dialogue model.
@All Right Reserve
'''

import os
import torch
import random
import traceback
import model.utils as utils
import model.dataset as dataset
from model.model_multi_input import MultiInputModel
from model.trainer_multi_input import Trainer
from model.text import Vocab
import argparse

class mylog:
    def info(self, text):
        print(text)

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file', default='config.json')
parser.add_argument('--gpu', help='which gpu to use', type=str, default='3')
parser.add_argument('--epoch', help='which epoch to use', type=int, default=-1)

args = parser.parse_args()
config = utils.load_config(args.config)
config_path = os.path.dirname(args.config)
train_dir = os.path.join(config_path, config['train_dir'])

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

try:
    print('pytorch version: {}'.format(torch.__version__))
    if args.epoch == -1:
        model_path = os.path.join(train_dir, utils.get_latest_ckpt(train_dir))
    else:
        model_path = os.path.join(train_dir, utils.get_ckpt_filename('model', args.epoch))

    if not os.path.isfile(model_path):
        print('cannot find {}'.format(model_path))
        exit(0)

    if len(args.gpu) != 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    vocab = Vocab(config.vocab_path)

    print('Building models')
    model =  MultiInputModel(config, vocab).to(device)

    print('Loading weights from {}'.format(model_path))
    state_dict = torch.load(model_path, map_location=device)['model']
    for i in list(state_dict.keys()):
        state_dict[i.replace('.module.', '.')] = state_dict.pop(i)
    model.load_state_dict(state_dict)
    model.eval()

    while True:
        post = input('>> ')
        post = ' '.join(list(post.replace(' ', '')))
        # print('post_str', post)
        post = [vocab.eos_id] + vocab.string2ids(post) + [vocab.eos_id]
        # print('post', post)
        contexts = [torch.tensor([post], dtype=torch.long, device=device)]
        # print('contexts', contexts)
        prediction = model.predict(contexts)[0]
        pred_str = vocab.ids2string(prediction)
        print('>> {}'.format(pred_str))

except:
    print(traceback.format_exc())
