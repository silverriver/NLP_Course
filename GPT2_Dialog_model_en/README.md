# Code for the GPT2 based NLG model (For English dialogue)

This is the implementation of the GPT2 based dialogue model. The multi-input scheme is implemented to condition the dialogue model on the dialogue context.

We have modified the `transformers` lib to implement the multi-input scheme.

## Usage

Please first download the pretrained GPT2 model. We suggest to use the [DialoGPT](https://huggingface.co/microsoft/DialoGPT-small) model.
Then please prepare the input data using the format demonstrated in the `data` folder.
Change the corresponding field in the `config.json` file to specify the location of GPT2 model and the data.

Use the following command to launch single-GPU training

    python3 train.py --config [path_to_config]

Use the following command to launch multi-GPT training

    bash run_training.sh
