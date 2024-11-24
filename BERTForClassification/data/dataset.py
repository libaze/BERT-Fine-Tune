# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/11/23 13:54
@File   : dataset.py
@desc   : 
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset

tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')


def collate_fn(batch):
    # print(batch)
    inputs = []
    labels = []
    for text, label in batch:
        inputs.append(text)
        labels.append(label)
    tokenized_text = tokenizer(inputs, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    return tokenized_text, torch.tensor(labels, dtype=torch.long)


class BertForClassificationDataset(Dataset):
    def __init__(self, data_path: str, split: str = 'train'):
        self.data = load_dataset(data_path)[split]
        # if split == 'train':
        #     self.data = self.data.select(range(20000))
        print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data['sentence'][idx]
        label = self.data['label'][idx]
        return sentence, label


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
    dataset = BertForClassificationDataset('./waimai_10k')
    print(len(dataset))
    print(dataset[0])
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    for batch in loader:
        print(batch)

