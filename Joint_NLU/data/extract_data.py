import os
import json
import random

in_file = '/home/data/tmp/NLP_Course/Joint_NLU/data/train.json'
out_train_file = '/home/data/tmp/NLP_Course/Joint_NLU/data/train.tsv'
out_test_file = '/home/data/tmp/NLP_Course/Joint_NLU/data/test.tsv'
cls_vocab_file = '/home/data/tmp/NLP_Course/Joint_NLU/data/cls_vocab'
slot_vocab_file = '/home/data/tmp/NLP_Course/Joint_NLU/data/slot_vocab'

with open(in_file) as f:
    data = json.load(f)

print('{} lines read'.format(len(data)))

cls_label = set()
slot_label = set()
total_data = []
for d in data:
    domain = d['domain']
    intent = d['intent']
    cls_label.add('{}@{}'.format(domain, intent))
    slots = ['o'] * len(d['text'])
    for s in d['slots']:
        idx = d['text'].index(d['slots'][s])
        slots[idx] = 'B-' + s

        for i in range(idx + 1, idx + len(d['slots'][s])):
            slots[i] = 'I-' + s
    slot_label.update(slots)
    total_data.append(('{}@{}\t{}\t{}'.format(domain, intent, d['text'], ' '.join(slots))).lower())
random.shuffle(total_data)

with open(out_train_file, 'w') as f:
    for i in total_data[:2000]:
        print(i, file=f)

with open(out_test_file, 'w') as f:
    for i in total_data[2000:]:
        print(i, file=f)
    
with open(cls_vocab_file, 'w') as f:
    for i in cls_label:
        print(i, file=f)

with open(slot_vocab_file, 'w') as f:
    for i in slot_label:
        print(i, file=f)

print('fin.')