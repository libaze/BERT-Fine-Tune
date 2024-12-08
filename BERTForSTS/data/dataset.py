# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/7/16 9:32
@File   : dataset.py
@desc   : 
"""
import json
import torch
import random
from datasets import Dataset as dt
from torch.utils.data import Dataset
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')


def collate_fn(batch):
    batch_s1 = []
    batch_s2 = []
    batch_labels = []
    for example in batch:
        batch_s1.append(example[0])
        batch_s2.append(example[1])
        batch_labels.append(example[2])
    tokenized_data = tokenizer(
        text=batch_s1,
        text_pair=batch_s2,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )
    labels = torch.tensor(batch_labels, dtype=torch.long)
    return tokenized_data, labels


class BERTForSTSDataset(Dataset):
    def __init__(self, data_path: str, split: str = 'train', select: int = 20000, random_seed: int = 666):
        super(BERTForSTSDataset, self).__init__()
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f.readlines()[:select]]
        random.seed(random_seed)
        random.shuffle(data)
        s1, s2 = int(select*0.7), int(select*0.8)
        if split == 'train':
            self.data = dt.from_list(data[:s1])
        elif split == 'test':
            self.data = dt.from_list(data[s2:])
        else:
            self.data = dt.from_list(data[s1:s2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence1 = self.data[idx]['sentence1']
        sentence2 = self.data[idx]['sentence2']
        label = self.data[idx]['label']
        return sentence1, sentence2, int(label)


if __name__ == '__main__':
    dataset = BERTForSTSDataset(data_path='./train_pair.json', select=20000, split='train')
    print(len(dataset))
    print(dataset[0])
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    for batch in loader:
        print(batch)



