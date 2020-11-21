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
                 log_dir, logger, device=torch.device('cuda'), valid_writer=None):
        self.config = args
        self.device = device
        self.logger = logger
        self.log_dir = log_dir
        self.tokz = tokz
        self.train_writer = SummaryWriter(os.path.join(log_dir, 'train_cls'))
        if valid_writer is None:
            self.valid_writer = SummaryWriter(os.path.join(log_dir, 'valid_cls'))
        else:
            self.valid_writer = valid_writer
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss().to(device)

        base_optimizer = Adam(self.model.parameters(), lr=self.config.lr, weight_decay=0.01)
        self.optimizer = NoamOpt(self.model.config.hidden_size, 0.1, self.config.lr_warmup, base_optimizer)

        self.train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=self.config.bs, num_workers=self.config.n_jobs, pin_memory=True,
            collate_fn=PadBatchSeq(self.tokz.pad_token_id))

        self.valid_dataloader = DataLoader(
            valid_dataset, shuffle=False, batch_size=self.config.bs, num_workers=self.config.n_jobs, pin_memory=True,
            collate_fn=PadBatchSeq(self.tokz.pad_token_id))

    def state_dict(self):
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _eval_train(self, epoch):
        self.model.train()

        loss, acc, step_count = 0, 0, 0
        total = len(self.train_dataloader)
        TQDM = tqdm(enumerate(self.train_dataloader), desc='Train (epoch #{})'.format(epoch),
                    dynamic_ncols=True, total=total)

        for i, data in TQDM:
            text, label = data['utt'].to(self.device), data['label'].to(self.device)
            mask = data['mask'].to(self.device)

            outputs = self.model(text, attention_mask=mask, return_dict=True)
            batch_loss = self.criterion(outputs.logits, label)
            batch_acc = (torch.argmax(outputs.logits, dim=1) == label).float().mean()

            full_loss = batch_loss / self.config.batch_split
            full_loss.backward()

            loss += batch_loss.item()
            acc += batch_acc.item()
            step_count += 1

            curr_step = self.optimizer.curr_step()
            lr = self.optimizer.param_groups[0]["lr"]
            # self.logger.info('epoch %d, batch %d' % (epoch, i))
            if (i + 1) % self.config.batch_split == 0:
                # update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss /= step_count
                acc /= step_count

                self.train_writer.add_scalar('ind/loss', loss, curr_step)
                self.train_writer.add_scalar('ind/acc', acc, curr_step)
                self.train_writer.add_scalar('ind/lr', lr, curr_step)
                TQDM.set_postfix({'loss': loss, 'acc': acc})

                loss, acc, step_count = 0, 0, 0

                # only valid on dev and sample on dev data at every eval_steps
                if curr_step % self.config.eval_steps == 0:
                    self._eval_test(epoch, curr_step)

    def _eval_test(self, epoch, step):
        self.model.eval()
        with torch.no_grad():
            all_logits = []
            all_label = []
            for d_data in self.valid_dataloader:
                text, label = d_data['utt'].to(self.device), d_data['label'].to(self.device)
                mask = d_data['mask'].to(self.device)
                outputs = self.model(text, attention_mask=mask, return_dict=True)
                all_label.append(label)
                all_logits.append(outputs.logits)

            all_logits = torch.cat(all_logits, dim=0)
            all_label = torch.cat(all_label, dim=0)

            loss = self.criterion(all_logits, all_label).float()
            acc = (torch.argmax(all_logits, dim=1) == all_label).float().mean()

            self.valid_writer.add_scalar('ind/loss', loss, step)
            self.valid_writer.add_scalar('ind/acc', acc, step)
            log_str = 'epoch {:>3}, step {}'.format(epoch, step)
            log_str += ', loss {:>4.4f}'.format(loss)
            log_str += ', acc {:>4.4f}'.format(acc)
            self.logger.info(log_str)

        self.model.train()


    def train(self, start_epoch, epochs, after_epoch_funcs=[], after_step_funcs=[]):
        for epoch in range(start_epoch + 1, epochs):
            self.logger.info('Training on epoch'.format(epoch))
            self._eval_train(epoch)
            for func in after_epoch_funcs:
                func(epoch, self.device)