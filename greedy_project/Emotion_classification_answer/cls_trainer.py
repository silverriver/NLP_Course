# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2020-09-20
@LastEditTime: 2020-09-20
@LastEditors: Please set LastEditors
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the training details of the emotion classifier.
@All Right Reserve
'''

from optim import Adam, NoamOpt
import torch
import os
import torch.nn as nn
import torch.distributed
import torch.tensor
from dataset import PadBatchSeq
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ClsTrainer:
    def __init__(self, args, model, tokz, train_dataset, valid_dataset,
                 log_dir, logger, device=torch.device('cuda'), valid_writer=None, distributed=False):
        # Initialize attributes
        self.config = args
        self.device = device
        self.logger = logger
        self.log_dir = log_dir
        self.tokz = tokz
        self.rank = torch.distributed.get_rank() if distributed else -1
        # Init the writer for tensorboard
        self.train_writer = SummaryWriter(os.path.join(log_dir, 'train_cls'))
        if valid_writer is None:
            self.valid_writer = SummaryWriter(os.path.join(log_dir, 'valid_cls'))
        else:
            self.valid_writer = valid_writer
        # Move the model to the right device
        self.model = model.to(device, non_blocking=True)
        # Init criterion 
        self.criterion = nn.CrossEntropyLoss().to(device)

        # Init optimizer. Note that we copied the implementation of Adaom from the original torch lib.
        base_optimizer = Adam(self.model.parameters(), lr=self.config.lr, weight_decay=0.01)
        # Set up optimizer for learning rate scheduling with warmup
        if hasattr(self.model, 'config'):
            self.optimizer = NoamOpt(self.model.config.hidden_size, 0.1, self.config.lr_warmup, base_optimizer)
        else:
            self.optimizer = NoamOpt(self.model.module.config.hidden_size, 0.1, self.config.lr_warmup, base_optimizer)

        # Init the sampler
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else torch.utils.data.RandomSampler(train_dataset)
        self.valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if distributed else None

        # Init dataloader
        self.train_dataloader = DataLoader(
            train_dataset, sampler=self.train_sampler, batch_size=self.config.bs, num_workers=self.config.n_jobs, pin_memory=True,
            collate_fn=PadBatchSeq(self.tokz.pad_token_id))

        self.valid_dataloader = DataLoader(
            valid_dataset, sampler=self.valid_sampler, batch_size=self.config.bs, num_workers=self.config.n_jobs, pin_memory=True,
            collate_fn=PadBatchSeq(self.tokz.pad_token_id))

    def state_dict(self):
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _eval_train(self, epoch):
        self.model.train()

        loss, acc, step_count = 0, 0, 0
        total = len(self.train_dataloader)
        if self.rank in [-1, 0]:
            TQDM = tqdm(enumerate(self.train_dataloader), desc='Train (epoch #{})'.format(epoch),
                        dynamic_ncols=True, total=total)
        else:
            TQDM = enumerate(self.train_dataloader)

        for i, data in TQDM:
            # Move input data to the right device
            text = data['utt'].to(self.device, non_blocking=True)
            label = data['label'].to(self.device, non_blocking=True)
            mask = data['mask'].to(self.device, non_blocking=True)

            # Forward pass
            _, logits = self.model(text, attention_mask=mask)
            
            # calculate loss. Note that the loss can also be calculated by passing the labels as input to the model.
            batch_loss = self.criterion(logits, label)
            batch_acc = (torch.argmax(logits, dim=1) == label).float().mean()

            # Rescale the loss
            full_loss = batch_loss / self.config.batch_split
            # Backward pass
            full_loss.backward()

            loss += batch_loss.item()
            acc += batch_acc.item()
            step_count += 1

            curr_step = self.optimizer.curr_step()
            lr = self.optimizer.param_groups[0]["lr"]
            if (i + 1) % self.config.batch_split == 0:
                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss /= step_count
                acc /= step_count

                if self.rank in [-1, 0]:
                    self.train_writer.add_scalar('ind/loss', loss, curr_step)
                    self.train_writer.add_scalar('ind/acc', acc, curr_step)
                    self.train_writer.add_scalar('ind/lr', lr, curr_step)
                    TQDM.set_postfix({'loss': loss, 'acc': acc})

                loss, acc, step_count = 0, 0, 0

                # Do evaluate of the model on the DEV set.
                if curr_step % self.config.eval_steps == 0:
                    self._eval_test(epoch, curr_step)

    def _eval_test(self, epoch, step):
        self.model.eval()
        with torch.no_grad():
            all_logits = []
            all_label = []
            for d_data in self.valid_dataloader:
                # Move input data to the right device
                text = d_data['utt'].to(self.device, non_blocking=True)
                label = d_data['label'].to(self.device, non_blocking=True)
                mask = d_data['mask'].to(self.device, non_blocking=True)
                # Forward pass
                _, logits = self.model(text, attention_mask=mask)
                all_label.append(label)
                all_logits.append(logits)

            all_logits = torch.cat(all_logits, dim=0)
            all_label = torch.cat(all_label, dim=0)

            # Calcuate the loss
            loss = self.criterion(all_logits, all_label).float()
            acc = (torch.argmax(all_logits, dim=1) == all_label).float().mean()

            if self.rank != -1:
                # In distrbuted training, the loss is summed and averaged across all the workers.
                torch.distributed.all_reduce(loss, op=torch.distributed.reduce_op.SUM)
                torch.distributed.all_reduce(acc, op=torch.distributed.reduce_op.SUM)

                loss /= torch.distributed.get_world_size()
                acc /= torch.distributed.get_world_size()

            if self.rank in [-1, 0]:
                # Writh the loss and accuracy to tensorboard and logs
                self.valid_writer.add_scalar('ind/loss', loss, step)
                self.valid_writer.add_scalar('ind/acc', acc, step)
                log_str = 'epoch {:>3}, step {}'.format(epoch, step)
                log_str += ', loss {:>4.4f}'.format(loss)
                log_str += ', acc {:>4.4f}'.format(acc)
                self.logger.info(log_str)

        self.model.train()

    def train(self, start_epoch, epochs, after_epoch_funcs=[], after_step_funcs=[]):
        # Train the model
        for epoch in range(start_epoch + 1, epochs):
            self.logger.info('Training on epoch'.format(epoch))
            # Reshuffle the train data in each epoch
            if hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)
            # Call the training function
            self._eval_train(epoch)
            for func in after_epoch_funcs:
                # Call the after epoch function
                func(epoch, self.device)
