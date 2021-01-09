# Code for joint NLU model

This implementation follows the work of [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/pdf/1902.10909.pdf).

## Usage

Use the script `trian.py` to launch the Single-GPU training:

    python3 train.py

Use the script `run_train.sh` to launch the Multi-GPT training:

    bash run_training.sh

## Data Processing

The data from the SMP2019 task is provided in the folder `data`.

## TODO
- Implement CRF layer. Code from this [repo](https://github.com/monologg/JointBERT/issues) can be referred to add the CRF layer.