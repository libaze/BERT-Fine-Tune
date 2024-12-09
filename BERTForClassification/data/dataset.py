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
    # 初始化输入和标签列表
    inputs = []
    labels = []
    # 遍历批次中的每个样本
    for text, label in batch:
        # 将文本添加到输入列表
        inputs.append(text)
        # 将标签添加到标签列表
        labels.append(label)
    # 使用tokenizer对输入文本进行编码
    # max_length指定最大长度，truncation进行截断，padding进行填充
    # return_tensors='pt'表示返回PyTorch张量
    tokenized_text = tokenizer(inputs, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    # 返回编码后的文本和标签的张量
    return tokenized_text, torch.tensor(labels, dtype=torch.long)


class BertForClassificationDataset(Dataset):
    def __init__(self, data_path: str, split: str = 'train'):
        # 初始化数据集，加载指定路径的数据集并选择分割（默认为'train'）
        self.data = load_dataset(data_path)[split]

    def __len__(self):
        # 返回数据集中的样本数量
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引idx获取数据集中的单个样本
        # 获取样本的句子
        sentence = self.data['sentence'][idx]
        # 获取样本的标签
        label = self.data['label'][idx]
        # 返回句子和对应的标签
        return sentence, label


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
    dataset = BertForClassificationDataset('./waimai_10k')
    print(len(dataset))
    print(dataset[0])
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    for batch in loader:
        print(batch)

