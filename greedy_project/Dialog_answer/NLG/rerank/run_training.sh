# Launch distributed training
python3 -m torch.distributed.launch --nproc_per_node=3  \
train.py --gpu '0,1,3' \