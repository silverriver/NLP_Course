# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2022-05-27
@LastEditTime: 2022-05-27
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implements a script that can infer output for a given set of data.
@All Right Reserve
'''

import os
import torch
import random
import traceback
import model.utils as utils
import model.dataset as dataset
from model.model_multi_input import MultiInputModel
from torch.utils.data import DataLoader
from model.text import Vocab
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file', default='infer_config.json')
parser.add_argument('--out_file', help='out_file', default='infer_out.txt')
parser.add_argument('--gpu', help='which gpu to use', type=str, default='2')
parser.add_argument("--local_rank", help='used for distributed training', type=int, default=-1)

args = parser.parse_args()
config = utils.load_config(args.config)
config_path = os.path.dirname(args.config)
logger = utils.get_logger(os.path.join(config_path, 'main.log'))

train_dir = os.path.join(config_path, config['train_dir'])
data_dir = os.path.join(config_path, config['data_dir'])
eval_dir = os.path.join(config_path, config['eval_dir'])
log_dir = os.path.join(config_path, config['log_dir'])
best_model = os.path.join(config_path, config['best_dir'])

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


try:
    logger.info('pytorch version: {}'.format(torch.__version__))
    for i in config:
        logger.info('{}: {}'.format(i, config[i]))
    for i in vars(args):
        logger.info('{}: {}'.format(i, getattr(args, i)))

    # code for distributed training
    distributed = (args.local_rank != -1)
    if distributed:
        print(args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.manual_seed(config.seed)
    else:
        device = torch.device("cuda", 0)

    vocab = Vocab(config.vocab_path)
    test_dataset = dataset.DialogDataset([os.path.join(data_dir, config.test_data)],
                                          vocab, logger, config.max_seq_len - 1)
    sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if distributed else None

    test_dataloader = DataLoader(test_dataset, sampler=sampler, pin_memory=True,
                                 batch_size=config.batch_size, collate_fn=dataset.PadBatchSeq(vocab.pad_id))

    logger.info('Building models')
    model =  MultiInputModel(config, vocab).to(device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    latest_ckpt = config.infer_ckpt
    logger.info('Weights loading from {}'.format(os.path.join(train_dir, latest_ckpt)))
    weights = torch.load(os.path.join(train_dir, latest_ckpt), map_location=device)['model']
    weight_keys = list(weights.keys())
    for key in weight_keys:
        if key.startswith('transformer_module.module'):
            weights['transformer_module' + key[len('transformer_module.module'):]] = weights[key]
            weights.pop(key)

    model.load_state_dict(weights, strict=True)

    with torch.no_grad():
        model.eval()
        res = []
        with open(os.path.join(os.path.dirname(args.out_file), os.path.basename(args.out_file) + str(args.local_rank)), 'w') as f:
            if args.local_rank == -1 or args.local_rank == 0:
                ITER = tqdm(test_dataloader, dynamic_ncols=True, total=len(test_dataloader))
            else:
                ITER = test_dataloader

            for data in ITER:
                prediction = model.predict_beam([data['post'].to(device)])
                bs = data['post'].shape[0]
                for i in range(bs):
                    post_str = data['post'][i].tolist()[1:]
                    post_str = vocab.ids2string(post_str[:post_str.index(vocab.eos_id)])
                    resp_str = data['resp'][i].tolist()[1:]
                    resp_str = vocab.ids2string(resp_str[:resp_str.index(vocab.eos_id)])
                    for j in prediction[i]:
                        pred_str = vocab.ids2string(j)
                        print('{}\t{}\t{}\t{}'.format(data['style'][i], post_str, pred_str, resp_str), file=f)

except:
    logger.error(traceback.format_exc())

