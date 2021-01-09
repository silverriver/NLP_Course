import sys
sys.path.append('..')
from modified_transformers import GPT2Tokenizer


class Tokenizer(GPT2Tokenizer):
    sp1_token = '[SP1]'
    sp2_token = '[SP2]'

    def __init__(self, **kwargs):
        super().__init__(
            bos_token='[BOS]',
            pad_token='[PAD]',
            sep_token='[SEP]',
            cls_token='[CLS]',
            **kwargs
        )
        self.add_special_tokens({"additional_special_tokens": [Tokenizer.sp1_token, Tokenizer.sp2_token]})

    def decode_with_no_eos(self, ids):
        res = []
        for i in ids:
            if i in [self.eos_token_id, self.pad_token_id]:
                break
            res.append(i)
        return self.decode(res)

    @property
    def sp1_token_id(self) -> int:
        return self.convert_tokens_to_ids(Tokenizer.sp1_token)

    @property
    def sp2_token_id(self) -> int:
        return self.convert_tokens_to_ids(Tokenizer.sp2_token)


if __name__ == '__main__':
    dialogpt_dir = '/home/data/dialogpt_small'
    tokz = Tokenizer.from_pretrained(dialogpt_dir)
    print(tokz.eos_token, tokz.eos_token_id)
    print(tokz.bos_token, tokz.bos_token_id)
    print(tokz.sp2_token, tokz.sp2_token_id)
    print(tokz.sep_token, tokz.sep_token_id)
    print(tokz.unk_token, tokz.unk_token_id)
    print(tokz.all_special_ids)
    print(tokz.all_special_tokens)
    print(tokz.additional_special_tokens_ids)
