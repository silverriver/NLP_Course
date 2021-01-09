import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file', default='config.json')
parser.add_argument('--gpu', help='which gpu to use', type=str, default='2')
parser.add_argument("--local_rank", help='used for distributed training', type=int, default=-1)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if True:
    import torch
    import traceback
    import model.utils as utils
    import model.dataset as dataset
    from model.model_multi_input import MultiInputModel
    from model.trainer_multi_input import Trainer
    from model.text import Tokenizer
    from torch.nn.parallel import DistributedDataParallel

config = utils.load_config(args.config)
config_path = os.path.dirname(args.config)
logger = utils.get_logger(os.path.join(config_path, 'main.log'))

train_dir = os.path.join(config_path, config['train_dir'])
eval_dir = os.path.join(config_path, config['eval_dir'])
log_dir = os.path.join(config_path, config['log_dir'])
best_model = os.path.join(config_path, config['best_dir'])

# helpers -----------------------------------------------------
def save_func(epoch, device):
    filename = utils.get_ckpt_filename('model', epoch)
    torch.save(trainer.state_dict(), os.path.join(train_dir, filename))
    if os.path.exists(os.path.join(train_dir, utils.get_ckpt_filename('model', epoch - 80))):
        os.remove(os.path.join(train_dir, utils.get_ckpt_filename('model', epoch - 80)))

# helpers -----------------------------------------------------


try:
    if args.local_rank == -1 or args.local_rank == 0:
        logger.info('pytorch version: {}'.format(torch.__version__))
        for i in config:
            logger.info('{}: {}'.format(i, config[i]))
        for i in vars(args):
            logger.info('{}: {}'.format(i, getattr(args, i)))

        dirs = [train_dir, eval_dir, log_dir, best_model]

        for d in dirs:
            if not os.path.isdir(d):
                logger.info('cannot find {}, mkdiring'.format(d))
                os.makedirs(d)

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

    tokz = Tokenizer.from_pretrained(config.dialogpt_dir)
    train_dataset = dataset.DialogDataset([config.dialog_train], tokz, logger, max_context_len=config.max_context_len, max_resp_len=config.max_resp_len)
    valid_dataset = dataset.DialogDataset([config.dialog_valid], tokz, logger, max_context_len=config.max_context_len, max_resp_len=config.max_resp_len)
    logger.info('Loading pretrained dialogpt model')

    model = MultiInputModel(config, tokz)
    logger.info('GPT weights loaded from {}'.format(config.dialogpt_dir))

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    trainer = Trainer(model, train_dataset, valid_dataset, config, log_dir, logger, device, ignore_idxs=tokz.all_special_ids,
                      distributed=distributed)

    if distributed:
        trainer.model.transformer_module = DistributedDataParallel(
            trainer.model.transformer_module, device_ids=[args.local_rank], output_device=args.local_rank)

    start_epoch = 0
    latest_ckpt = utils.get_latest_ckpt(train_dir)
    if latest_ckpt is not None:
        logger.info('Weights loading from {}'.format(os.path.join(train_dir, latest_ckpt)))
        start_epoch = utils.get_epoch_from_ckpt(latest_ckpt)
        trainer.load_state_dict(torch.load(os.path.join(train_dir, latest_ckpt), map_location=device))

    try:
        if args.local_rank in [-1, 0]:
            trainer.train(start_epoch, config.n_epochs, after_epoch_funcs=[save_func])
        else:
            trainer.train(start_epoch, config.n_epochs)
    except (KeyboardInterrupt, Exception) as e:
        torch.save(trainer.state_dict(), os.path.join(train_dir, 'interrupt.pt'))
        raise e
except BaseException:
    logger.error(traceback.format_exc())
