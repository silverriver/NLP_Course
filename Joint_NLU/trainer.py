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


class Trainer:
    def __init__(self, args, model, tokz, train_dataset, valid_dataset,
                 log_dir, logger, device=torch.device('cuda'), valid_writer=None, distributed=False):
        self.config = args
        self.device = device
        self.logger = logger
        self.log_dir = log_dir
        self.tokz = tokz
        self.rank = torch.distributed.get_rank() if distributed else -1
        self.train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
        if valid_writer is None:
            self.valid_writer = SummaryWriter(os.path.join(log_dir, 'valid'))
        else:
            self.valid_writer = valid_writer
        self.model = model.to(device, non_blocking=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokz.pad_token_id, reduction='none').to(device)

        base_optimizer = Adam(self.model.parameters(), lr=self.config.lr, weight_decay=0.01)
        if hasattr(self.model, 'config'):
            self.optimizer = NoamOpt(self.model.config.hidden_size, 0.1, self.config.lr_warmup, base_optimizer)
        else:
            self.optimizer = NoamOpt(self.model.module.config.hidden_size, 0.1, self.config.lr_warmup, base_optimizer)

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else torch.utils.data.RandomSampler(train_dataset)
        self.valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if distributed else None

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

        intent_loss, slot_loss, intent_acc, slot_acc, step_count = 0, 0, 0, 0, 0
        total = len(self.train_dataloader)
        if self.rank in [-1, 0]:
            TQDM = tqdm(enumerate(self.train_dataloader), desc='Train (epoch #{})'.format(epoch),
                        dynamic_ncols=True, total=total)
        else:
            TQDM = enumerate(self.train_dataloader)

        for i, data in TQDM:
            text = data['utt'].to(self.device, non_blocking=True)
            intent_labels = data['intent'].to(self.device, non_blocking=True)
            slot_labels = data['slot'].to(self.device, non_blocking=True)
            mask = data['mask'].to(self.device, non_blocking=True)
            token_type = data['token_type'].to(self.device, non_blocking=True)

            intent_logits, slot_logits = self.model(input_ids=text, attention_mask=mask, token_type_ids=token_type)

            batch_intent_loss = self.criterion(intent_logits, intent_labels).mean()
            batch_slot_loss = self.criterion(slot_logits.view(-1, slot_logits.shape[-1]), slot_labels.view(-1)).mean()
            slot_mask = 1 - slot_labels.eq(self.tokz.pad_token_id).float()
            batch_slot_loss = (batch_slot_loss * slot_mask.view(-1)).sum() / slot_mask.sum()

            batch_loss = batch_intent_loss + batch_slot_loss
            batch_intent_acc = (torch.argmax(intent_logits, dim=-1) == intent_labels).float().mean()
            batch_slot_acc = (torch.argmax(slot_logits, dim=-1) == slot_labels)
            batch_slot_acc = torch.sum(batch_slot_acc * slot_mask) / torch.sum(slot_mask)

            full_loss = batch_loss / self.config.batch_split
            full_loss.backward()

            intent_loss += batch_intent_loss.item()
            slot_loss += batch_slot_loss.item()
            intent_acc += batch_intent_acc.item()
            slot_acc += batch_slot_acc.item()
            step_count += 1

            curr_step = self.optimizer.curr_step()
            lr = self.optimizer.param_groups[0]["lr"]
            # self.logger.info('epoch %d, batch %d' % (epoch, i))
            if (i + 1) % self.config.batch_split == 0:
                # update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

                intent_loss /= step_count
                slot_loss /= step_count
                intent_acc /= step_count
                slot_acc /= step_count

                if self.rank in [-1, 0]:
                    self.train_writer.add_scalar('loss/intent_loss', intent_loss, curr_step)
                    self.train_writer.add_scalar('loss/slot_loss', slot_loss, curr_step)
                    self.train_writer.add_scalar('acc/intent_acc', intent_acc, curr_step)
                    self.train_writer.add_scalar('acc/slot_acc', slot_acc, curr_step)
                    self.train_writer.add_scalar('lr', lr, curr_step)
                    TQDM.set_postfix({'intent_loss': intent_loss, 'intent_acc': intent_acc, 'slot_loss': slot_loss, 'slot_acc': slot_acc})

                intent_loss, slot_loss, intent_acc, slot_acc, step_count = 0, 0, 0, 0, 0

                # only valid on dev and sample on dev data at every eval_steps
                if curr_step % self.config.eval_steps == 0:
                    self._eval_test(epoch, curr_step)

    def _eval_test(self, epoch, step):
        self.model.eval()
        with torch.no_grad():
            dev_intent_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            dev_slot_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            dev_intent_acc = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            dev_slot_acc = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            count = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            for data in self.valid_dataloader:
                text = data['utt'].to(self.device, non_blocking=True)
                intent_labels = data['intent'].to(self.device, non_blocking=True)
                slot_labels = data['slot'].to(self.device, non_blocking=True)
                mask = data['mask'].to(self.device, non_blocking=True)
                token_type = data['token_type'].to(self.device, non_blocking=True)

                intent_logits, slot_logits = self.model(input_ids=text, attention_mask=mask, token_type_ids=token_type)
                
                batch_intent_loss = self.criterion(intent_logits, intent_labels)
                batch_slot_loss = self.criterion(slot_logits.view(-1, slot_logits.shape[-1]), slot_labels.view(-1))
                slot_mask = 1 - slot_labels.eq(self.tokz.pad_token_id).float()
                batch_slot_loss = (batch_slot_loss * slot_mask.view(-1)).view(text.shape[0], -1).sum(dim=-1) / slot_mask.sum(dim=-1)
                
                dev_intent_loss += batch_intent_loss.sum()
                dev_slot_loss += batch_slot_loss.sum()

                batch_intent_acc = (torch.argmax(intent_logits, dim=-1) == intent_labels).sum()
                batch_slot_acc = (torch.argmax(slot_logits, dim=-1) == slot_labels)
                batch_slot_acc = torch.sum(batch_slot_acc * slot_mask, dim=-1) / torch.sum(slot_mask, dim=-1)

                dev_intent_acc += batch_intent_acc
                dev_slot_acc += batch_slot_acc.sum()
                count += text.shape[0]

            if self.rank != -1:
                torch.distributed.all_reduce(dev_intent_loss, op=torch.distributed.reduce_op.SUM)
                torch.distributed.all_reduce(dev_slot_loss, op=torch.distributed.reduce_op.SUM)
                torch.distributed.all_reduce(dev_intent_acc, op=torch.distributed.reduce_op.SUM)
                torch.distributed.all_reduce(dev_slot_acc, op=torch.distributed.reduce_op.SUM)
                torch.distributed.all_reduce(count, op=torch.distributed.reduce_op.SUM)

            dev_intent_loss /= count
            dev_slot_loss /= count
            dev_intent_acc /= count
            dev_slot_acc /= count

            if self.rank in [-1, 0]:
                self.valid_writer.add_scalar('loss/intent_loss', dev_intent_loss, step)
                self.valid_writer.add_scalar('loss/slot_loss', dev_slot_loss, step)
                self.valid_writer.add_scalar('acc/intent_acc', dev_intent_acc, step)
                self.valid_writer.add_scalar('acc/slot_acc', dev_slot_acc, step)
                log_str = 'epoch {:>3}, step {}'.format(epoch, step)
                log_str += ', dev_intent_loss {:>4.4f}'.format(dev_intent_loss)
                log_str += ', dev_slot_loss {:>4.4f}'.format(dev_slot_loss)
                log_str += ', dev_intent_acc {:>4.4f}'.format(dev_intent_acc)
                log_str += ', dev_slot_acc {:>4.4f}'.format(dev_slot_acc)
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