import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--bert_path', help='config file', default='/home/data/tmp/bert-base-chinese')
parser.add_argument('--save_path', help='path to save checkpoints', default='/home/data/tmp/NLP_Course/Joint_NLU/train')
parser.add_argument('--train_file', help='training data', default='/home/data/tmp/NLP_Course/Joint_NLU/data/train.tsv')
parser.add_argument('--valid_file', help='valid data', default='/home/data/tmp/NLP_Course/Joint_NLU/data/test.tsv')
parser.add_argument('--intent_label_vocab', help='training file', default='/home/data/tmp/NLP_Course/Joint_NLU/data/cls_vocab')
parser.add_argument('--slot_label_vocab', help='training file', default='/home/data/tmp/NLP_Course/Joint_NLU/data/slot_vocab')

parser.add_argument("--local_rank", help='used for distributed training', type=int, default=-1)
parser.add_argument('--lr', type=float, default=8e-6)
parser.add_argument('--lr_warmup', type=float, default=200)
parser.add_argument('--bs', type=int, default=30)
parser.add_argument('--batch_split', type=int, default=1)
parser.add_argument('--eval_steps', type=int, default=40)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--max_length', type=int, default=90)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--n_jobs', type=int, default=1, help='num of workers to process data')

parser.add_argument('--gpu', help='which gpu to use', type=str, default='3')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from transformers import BertConfig, BertTokenizer, AdamW
from NLU_model import NLUModule
import dataset
import utils
import traceback
from trainer import Trainer
from torch.nn.parallel import DistributedDataParallel

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
        device = torch.device("cuda", 0)
    tokz = BertTokenizer.from_pretrained(args.bert_path)
    _, intent2index, _ = utils.load_vocab(args.intent_label_vocab)
    _, slot2index, _ = utils.load_vocab(args.slot_label_vocab)
    train_dataset = dataset.NLUDataset([args.train_file], tokz, intent2index, slot2index, logger, max_lengths=args.max_length)
    valid_dataset = dataset.NLUDataset([args.valid_file], tokz, intent2index, slot2index, logger, max_lengths=args.max_length)

    logger.info('Building models, rank {}'.format(args.local_rank))
    bert_config = BertConfig.from_pretrained(args.bert_path)
    bert_config.num_intent_labels = len(intent2index)
    bert_config.num_slot_labels = len(slot2index)
    model = NLUModule.from_pretrained(args.bert_path, config=bert_config).to(device)

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
