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


# 加载预训练的BERT分词器
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')

# 定义一个批处理函数，用于将数据转换为模型所需的格式
def collate_fn(batch):
    batch_s1 = []  # 初始化句子1列表
    batch_s2 = []  # 初始化句子2列表
    batch_labels = []  # 初始化标签列表
    for example in batch:  # 遍历每个样本
        batch_s1.append(example[0])  # 添加句子1
        batch_s2.append(example[1])  # 添加句子2
        batch_labels.append(example[2])  # 添加标签
    # 使用tokenizer对句子进行编码，设置最大长度为128，进行截断和填充
    tokenized_data = tokenizer(
        text=batch_s1,
        text_pair=batch_s2,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors="pt"  # 返回PyTorch张量
    )
    labels = torch.tensor(batch_labels, dtype=torch.long)  # 将标签转换为PyTorch长整型张量
    return tokenized_data, labels  # 返回编码后的数据和标签


class BERTForSTSDataset(Dataset):
    def __init__(self, data_path: str, split: str = 'train', select: int = 20000, random_seed: int = 666):
        super(BERTForSTSDataset, self).__init__()  # 调用父类初始化方法
        with open(data_path, 'r', encoding='utf-8') as f:  # 打开数据文件
            data = [json.loads(line.strip()) for line in f.readlines()[:select]]  # 读取并解析前select行数据
        random.seed(random_seed)  # 设置随机种子
        random.shuffle(data)  # 打乱数据顺序
        s1, s2 = int(select*0.7), int(select*0.8)  # 计算训练集、验证集和测试集的划分点
        if split == 'train':  # 如果是训练集
            self.data = data[:s1]  # 使用前70%的数据
        elif split == 'test':  # 如果是测试集
            self.data = data[s2:]  # 使用最后20%的数据
        else:  # 否则是验证集
            self.data = data[s1:s2]  # 使用中间10%的数据

    def __len__(self):
        # 返回数据的长度
        return len(self.data)

    def __getitem__(self, idx):
        sentence1 = self.data[idx]['sentence1']  # 获取句子1
        sentence2 = self.data[idx]['sentence2']  # 获取句子2
        label = self.data[idx]['label']  # 获取标签
        return sentence1, sentence2, int(label)  # 返回句子对和对应的标签


if __name__ == '__main__':
    dataset = BERTForSTSDataset(data_path='./train_pair.json', select=20000, split='train')
    print(len(dataset))
    print(dataset[0])
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    for batch in loader:
        print(batch)



