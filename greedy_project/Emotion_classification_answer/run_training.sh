python3 -m torch.distributed.launch --nproc_per_node=3  \
train.py --gpu '1,2,3' \