# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2022-05-27
@LastEditTime: 2022-05-27
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the training script.
@All Right Reserve
'''

import torch
import os
import random
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import math
import torch.tensor
from .dataset import PadBatchSeq
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .optim import Adam, NoamOpt
from .loss import LabelSmoothingLoss


class Trainer:
    def __init__(self, model, train_dataset, valid_dataset, config, log_dir, logger, device=torch.device('cuda'),
                 ignore_idxs=[], distributed=False):
        self.config = config
        self.device = device
        self.logger = logger
        self.log_dir = log_dir
        self.valid_dataset = valid_dataset
        self.rank = torch.distributed.get_rank() if distributed else -1
        self.train_writer = SummaryWriter(os.path.join(log_dir, 'train'), flush_secs=60)
        self.valid_writer = SummaryWriter(os.path.join(log_dir, 'valid'))
        self.ignore_idxs = ignore_idxs
        self.model = model.to(device)
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=self.model.vocab.pad_id).to(device)
        self.criterion = LabelSmoothingLoss(n_labels=len(self.model.vocab), smoothing=config.label_smoothing,
                                            ignore_index=self.model.vocab.pad_id).to(device)
        base_optimizer = Adam(self.model.parameters(), lr=config.lr, weight_decay=0.01)
        self.optimizer = NoamOpt(self.model.config.embeddings_size, 0.1, config.lr_warmup, base_optimizer)

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if distributed else None

        self.train_dataloader = DataLoader(train_dataset, sampler=self.train_sampler, batch_size=config.batch_size,
                                           num_workers=config.n_jobs, pin_memory=True,
                                           collate_fn=PadBatchSeq(self.model.vocab.pad_id))
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, sampler=self.valid_sampler,
                                           num_workers=config.n_jobs, pin_memory=True,
                                           collate_fn=PadBatchSeq(self.model.vocab.pad_id))

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=True)
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _eval_train(self, epoch):
        self.model.train()
        logged_step = -1
        loss = 0
        lm_loss = 0
        log_lm_loss, log_s2s_loss, step_count = 0, 0, 0
        total = len(self.train_dataloader)
        if self.rank == -1 or self.rank == 0:
            ITER = tqdm(enumerate(self.train_dataloader), dynamic_ncols=True, total=total)
        else:
            ITER = enumerate(self.train_dataloader)

        for i, data in ITER:
            post, resp = data['post'].to(self.device), data['resp'].to(self.device)
            enc_contexts = []

            # lm loss
            post_rep = self.model.encode(post.clone())
            enc_contexts.append(post_rep)

            post_outputs = self.model.generate(post_rep[0])
            ignore_mask = torch.stack([post == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1).bool()
            post.masked_fill_(ignore_mask, self.model.vocab.pad_id)
            prevs, nexts = post_outputs[:, :-1, :].contiguous(), post[:, 1:].contiguous()
            batch_lm_loss = self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1))

            # s2s loss
            prevs, nexts = resp[:, :-1].contiguous(), resp[:, 1:].contiguous()
            outputs = self.model.decode(prevs, enc_contexts)
            outputs = F.log_softmax(outputs, dim=-1)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            # optimization
            full_loss = (batch_lm_loss * self.config.lm_weight + batch_loss) / self.config.batch_split
            full_loss.backward()

            lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1)
            loss = (i * loss + batch_loss.item()) / (i + 1)

            log_lm_loss += batch_lm_loss.item()
            log_s2s_loss += batch_loss.item()
            step_count += 1

            if (i + 1) % self.config.batch_split == 0:
                if self.config.clip_grad is not None:
                    for group in self.optimizer.param_groups:
                        nn.utils.clip_grad_norm_(group['params'], self.config.clip_grad)
                # update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

                # shit log if you are node 0 in every step
                if self.rank == -1 or self.rank == 0:
                    log_lm_loss /= step_count
                    log_s2s_loss /= step_count
                    self.train_writer.add_scalar('loss/lm_loss', log_lm_loss, self.optimizer.curr_step())
                    self.train_writer.add_scalar('loss/s2s_loss', log_s2s_loss, self.optimizer.curr_step())
                    self.train_writer.add_scalar('ppl/s2s_loss', math.exp(log_s2s_loss), self.optimizer.curr_step())
                    self.train_writer.add_scalar('loss/total_loss', log_lm_loss + log_s2s_loss,
                                                 self.optimizer.curr_step())
                    self.train_writer.add_scalar('lr/lr', self.optimizer.rate(), self.optimizer.curr_step())
                    log_lm_loss, log_s2s_loss, step_count = 0, 0, 0

                # only valid on dev and sample on dev data at every eval_steps
                if self.optimizer.curr_step() % self.config.eval_steps == 0:
                    valid_lm_loss, valid_s2s_loss = self._eval_test()
                    if self.rank != -1:
                        torch.distributed.all_reduce(valid_lm_loss, op=torch.distributed.reduce_op.SUM)
                        torch.distributed.all_reduce(valid_s2s_loss, op=torch.distributed.reduce_op.SUM)
                        # self.logger.info("Reduced on rank {}, {}, {}".format(self.rank, valid_lm_loss.item(), valid_s2s_loss.item()))
                        valid_lm_loss /= torch.distributed.get_world_size()
                        valid_s2s_loss /= torch.distributed.get_world_size()

                    # but only shit log if you are node 0
                    if self.rank == -1 or self.rank == 0:
                        valid_lm_loss = valid_lm_loss.item()
                        valid_s2s_loss = valid_s2s_loss.item()
                        self.valid_writer.add_scalar('loss/lm_loss', valid_lm_loss, self.optimizer.curr_step())
                        self.valid_writer.add_scalar('loss/s2s_loss', valid_s2s_loss, self.optimizer.curr_step())
                        self.valid_writer.add_scalar('ppl/s2s_loss', math.exp(valid_s2s_loss), self.optimizer.curr_step())
                        self.valid_writer.add_scalar(
                            'loss/total_loss', valid_s2s_loss + valid_lm_loss, self.optimizer.curr_step())

                        log_str = ('epoch {:>3}, t_lm_loss {:>4.4f}, t_s2s_loss {:>4.4f}, ' +
                                   'v_lm_loss {:>4.4f}, v_s2s_loss {:>4.4f} lr {:>.6}, step {}').format(
                            epoch, lm_loss, loss, valid_lm_loss, valid_s2s_loss, self.optimizer.rate(),
                            self.optimizer.curr_step())
                        self.logger.info(log_str)

                        # and only predicts sample on node 0
                        sample_dialog = self._pred_sample(5)
                        for j, d in enumerate(sample_dialog):
                            self.logger.info('--epoch {} step{} sample {}--'.format(
                                epoch, self.optimizer.curr_step(), j))
                            self.logger.info('post: {}'.format(d['post']))
                            self.logger.info('resp: {}'.format(d['resp']))
                            self.logger.info('pred: {}'.format(d['pred']))
                            self.train_writer.add_text('dialog', 'Post: {}\n  Resp: {}\n  Pred: {}\n'.format(
                                d['post'], d['resp'], d['pred']), self.optimizer.curr_step())
                    self.model.train()

    def _eval_test(self):
        '''
        Task4
        This function is used to evaluate the model on dev data.
        return:
            lm_loss: loss on lm data
            s2s_loss: loss on s2s data

        This function should be implemented as follows:
        1. set the model to eval mode
        2. iterate through the dev data
        3. for each batch, fist pass the post and resp in GPU
        4. calculate the lm loss and the s2s loss. (you can refer to the implement in the train function)
        5. accumlate the lm loss and s2s loss for all batches and return the average lm loss and s2s loss
        '''
        ###############################
        # YOUR CODE HERE for Task 4   #
        ###############################
        raise NotImplementedError

    def _pred_sample(self, n_sample):
        with torch.no_grad():
            self.model.eval()
            samples_idxs = random.sample(range(len(self.valid_dataset)), n_sample)
            samples = PadBatchSeq(self.model.vocab.pad_id)([self.valid_dataset[idx] for idx in samples_idxs])
            prediction = self.model.predict([samples['post'].to(self.device)])
            res = []
            for j in range(len(samples_idxs)):
                post_str = samples['post'][j].tolist()[1:]
                post_str = self.model.vocab.ids2string(post_str[:post_str.index(self.model.vocab.eos_id)])
                resp_str = samples['resp'][j].tolist()[1:]
                resp_str = self.model.vocab.ids2string(resp_str[:resp_str.index(self.model.vocab.eos_id)])
                pred_str = self.model.vocab.ids2string(prediction[j])
                res.append({"post": post_str, "resp": resp_str, "pred": pred_str})

        return res

    def test(self):
        self._eval_test()

    def train(self, start_epoch, epochs, after_epoch_funcs=[]):
        for epoch in range(start_epoch + 1, epochs):
            self.logger.info('Training on process {}, epoch {}, step {}'.format(
                self.rank, epoch, self.optimizer.curr_step()))
            if self.train_sampler and hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)
            self._eval_train(epoch)
            # if epoch % 10 == 0 and epoch > 0:
            for func in after_epoch_funcs:
                func(epoch, self.device)
