# Launch distributed training
python3 -m torch.distributed.launch --nproc_per_node=2  \
train.py --gpu '1,3' \