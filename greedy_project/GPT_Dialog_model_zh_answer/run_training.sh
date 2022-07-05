#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=3  \
train.py \
--config config.json \
--gpu '2,5,6' \
