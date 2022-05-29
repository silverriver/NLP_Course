import os
import json
import random

with open('raw_train.json') as f:
    res = json.load(f)

print(f'len(res): {len(res)}')
slot_vocab = set()
token_vocab = dict()
intent_vocab = set()
intent_vocab.add('chat_intent')
slot_vocab.add('o')

with open('ECDT2019.txt', 'w') as f:
    for l in res:
        utt = l['text'].strip().replace(' ', '')
        intent = l['intent']
        domain = l['domain']
        intent_vocab.add(intent)
        slots = ['o'] * len(utt)
        for w in utt:
            if w not in token_vocab:
                token_vocab[w] = 0
            token_vocab[w] += 1
            
        for key, value in l['slots'].items():
            index = utt.find(value)
            for j in range(index, index + len(value)):
                if slots[j] != 'o':
                    print(l)
                    break
                if j == index:
                    slots[j] = 'b-' + key
                    slot_vocab.add(slots[j])
                else:
                    slots[j] = 'i-' + key
                    slot_vocab.add(slots[j])
        print(f'{domain}\t{intent}\t{" ".join(list(utt))}\t{" ".join(slots)}', file=f)

with open('intent_vocab.txt', 'w') as f:
    for intent in intent_vocab:
        print(intent, file=f)

with open('slot_vocab.txt', 'w') as f:
    for slot in slot_vocab:
        print(slot, file=f)

token_vocab = sorted([(k, v) for k, v in token_vocab.items()], key=lambda x: x[1], reverse=True)

with open('word_vocab.txt', 'w') as f:
    for token, freq in token_vocab:
        print(token, file=f)

data = []
with open('ECDT2019.txt', 'r') as f:
    for line in f:
        data.append(line.strip())

with open('open_domain.txt', 'r') as f:
    for line in f:
        data.append(line.strip())

random.shuffle(data)
valid = int(len(data) * 0.2)

with open('train.txt', 'w', encoding='utf8') as f:
    for line in data[:-valid]:
        print(line, file=f)

with open('dev.txt', 'w', encoding='utf8') as f:
    for line in data[-valid:]:
        print(line, file=f)

print('fin.')
