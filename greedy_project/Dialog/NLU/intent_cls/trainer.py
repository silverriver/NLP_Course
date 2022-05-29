# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2022-05-27
@LastEditTime: 2022-05-27
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the Intent Classification model.
@All Right Reserve
'''

from transformers import AdamW, get_cosine_schedule_with_warmup
import torch
import os
import torch.nn as nn
import torch.distributed
from dataset import PadBatchSeq
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F


class KDLoss:
    def __init__(self, alpha, temperature):
        self.alpha = alpha
        self.T = temperature
    
    def __call__(self, outputs, labels, teacher_outputs):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities!
        :param outputs (Tensor): student's output logits
        :param labels (Tensor): student's target labels
        :param teacher_outputs (Tensor): teacher's output logits
        """
        #############################################################
        # 5. TODO complete the code for knowledge distillation      #
        # follow the following steps:                               #  
        # 1. Call torch.nn.KLDivLoss to calculate distillation loss #
        #    follow the paper https://arxiv.org/abs/1503.02531      #
        # 2. Call torch.nn.CrossEntropyLoss to calculate CE loss    #
        # 3. Add the two losses together                            #
        # 4. return                                                 #
        #############################################################
        pass 

class Trainer:
    def __init__(self, args, model, teacher_model, tokz, train_dataset, valid_dataset,
                 log_dir, logger, device=torch.device('cuda'), valid_writer=None, distributed=False):
        self.config = args
        self.device = device
        self.logger = logger
        self.log_dir = log_dir
        self.tokz = tokz
        self.teacher_model = teacher_model
        self.rank = torch.distributed.get_rank() if distributed else -1
        self.train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
        if valid_writer is None:
            self.valid_writer = SummaryWriter(os.path.join(log_dir, 'valid'))
        else:
            self.valid_writer = valid_writer
        self.model = model.to(device, non_blocking=True)
        if self.teacher_model is None:
            self.criterion = nn.CrossEntropyLoss(ignore_index=tokz.pad_token_id, reduction='none').to(device)
        else:
            self.criterion = KDLoss(alpha=0.5, temperature=0.4)  # You can tune the hyper-parameters here.

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else torch.utils.data.RandomSampler(train_dataset)
        self.valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if distributed else None

        self.train_dataloader = DataLoader(
            train_dataset, sampler=self.train_sampler, batch_size=self.config.bs, num_workers=self.config.n_jobs, pin_memory=True,
            collate_fn=PadBatchSeq(self.tokz.pad_token_id))

        self.valid_dataloader = DataLoader(
            valid_dataset, sampler=self.valid_sampler, batch_size=self.config.bs, num_workers=self.config.n_jobs, pin_memory=True,
            collate_fn=PadBatchSeq(self.tokz.pad_token_id))

        self.optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        self.lr_scheduler  = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=len(self.train_dataloader) * self.config.n_epochs // self.config.batch_split)

    def state_dict(self):
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _eval_train(self, epoch):
        self.model.train()

        intent_loss, intent_acc, step_count = 0, 0, 0
        total = len(self.train_dataloader)
        if self.rank in [-1, 0]:
            TQDM = tqdm(enumerate(self.train_dataloader), desc='Train (epoch #{})'.format(epoch),
                        dynamic_ncols=True, total=total)
        else:
            TQDM = enumerate(self.train_dataloader)

        for i, data in TQDM:
            text = data['utt'].to(self.device, non_blocking=True)
            intent_labels = data['intent'].to(self.device, non_blocking=True)
            mask = data['mask'].to(self.device, non_blocking=True)
            token_type = data['token_type'].to(self.device, non_blocking=True)

            intent_logits = self.model(input_ids=text, attention_mask=mask, token_type_ids=token_type)
            if self.teacher_model is not None:
                teacher_logits = self.teacher_model(input_ids=text, attention_mask=mask, token_type_ids=token_type)
                batch_loss = self.criterion(intent_logits, intent_labels, teacher_logits).mean()
            else:
                batch_loss = self.criterion(intent_logits, intent_labels).mean()
            batch_intent_acc = (torch.argmax(intent_logits, dim=-1) == intent_labels).float().mean()

            full_loss = batch_loss / self.config.batch_split
            full_loss.backward()

            intent_loss += batch_loss.item()
            intent_acc += batch_intent_acc.item()
            step_count += 1

            self.lr_scheduler.state_dict
            if (i + 1) % self.config.batch_split == 0:
                # update weights
                self.lr_scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()

                intent_loss /= step_count
                intent_acc /= step_count
                curr_step = self.optimizer.state[self.optimizer.param_groups[0]["params"][-1]]["step"]

                if self.rank in [-1, 0]:
                    self.train_writer.add_scalar('loss/intent_loss', intent_loss, curr_step)
                    self.train_writer.add_scalar('acc/intent_acc', intent_acc, curr_step)
                    self.train_writer.add_scalar('lr', self.lr_scheduler.get_last_lr()[0], curr_step)
                    TQDM.set_postfix({'intent_loss': intent_loss, 'intent_acc': intent_acc})

                intent_loss, intent_acc, step_count = 0, 0, 0

                # only valid on dev and sample on dev data at every eval_steps
                if curr_step % self.config.eval_steps == 0:
                    self._eval_test(epoch, curr_step)

    def _eval_test(self, epoch, step):
        self.model.eval()
        with torch.no_grad():
            dev_intent_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            dev_intent_acc = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            count = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            for data in self.valid_dataloader:
                text = data['utt'].to(self.device, non_blocking=True)
                intent_labels = data['intent'].to(self.device, non_blocking=True)
                mask = data['mask'].to(self.device, non_blocking=True)
                token_type = data['token_type'].to(self.device, non_blocking=True)

                intent_logits = self.model(input_ids=text, attention_mask=mask, token_type_ids=token_type)

                if self.teacher_model is not None:
                    teacher_logits = self.teacher_model(input_ids=text, attention_mask=mask, token_type_ids=token_type)
                    loss = self.criterion(intent_logits, intent_labels, teacher_logits).sum()
                else:
                    loss = self.criterion(intent_logits, intent_labels).sum()

                dev_intent_loss += loss
                
                batch_intent_acc = (torch.argmax(intent_logits, dim=-1) == intent_labels).sum()

                dev_intent_acc += batch_intent_acc
                count += text.shape[0]

            if self.rank != -1:
                torch.distributed.all_reduce(dev_intent_loss, op=torch.distributed.reduce_op.SUM)
                torch.distributed.all_reduce(dev_intent_acc, op=torch.distributed.reduce_op.SUM)
                torch.distributed.all_reduce(count, op=torch.distributed.reduce_op.SUM)

            dev_intent_loss /= count
            dev_intent_acc /= count

            if self.rank in [-1, 0]:
                self.valid_writer.add_scalar('loss/intent_loss', dev_intent_loss.item(), step)
                self.valid_writer.add_scalar('acc/intent_acc', dev_intent_acc.item(), step)
                log_str = 'epoch {:>3}, step {}'.format(epoch, step)
                log_str += ', dev_intent_loss {:>4.4f}'.format(dev_intent_loss.item())
                log_str += ', dev_intent_acc {:>4.4f}'.format(dev_intent_acc.item())
                self.logger.info(log_str)

        self.model.train()


    def train(self, start_epoch, epochs, after_epoch_funcs=[], after_step_funcs=[]):
        for epoch in range(start_epoch + 1, epochs):
            self.logger.info('Training on epoch'.format(epoch))
            if hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)
            self._eval_train(epoch)
            for func in after_epoch_funcs:
                func(epoch, self.device)
