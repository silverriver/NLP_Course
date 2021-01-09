# Code for the GPT based NLG model (For Chinese dialogue)

This is the implementation of the GPT based dialogue model. The multi-input scheme is implemented to condition the dialogue model on the dialogue context.

The code is modified from this [repo](https://github.com/atselousov/transformer_chatbot)

## Usage

Please first download the pretrained GPT model from [here](https://pan.baidu.com/s/1l_jLVcpBnGXpLp7yf3lqiw) (extract code: `nmoc`).
Then please prepare the input data using the format demonstrated in the `data` folder.
Change the corresponding field in the `config.json` file to specify the location of GPT model and the data.

Use the following command to launch single-GPU training

    python3 train.py --config [path_to_config]

Use the following command to launch multi-GPT training

    bash run_training.sh
