# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/7/16 9:32
@File   : dataset.py
@desc   : 
"""
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizer

# 加载预训练的BERT分词器
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')


def collate_fn(batch):
    batch_tokens = []  # 初始化批次中的所有文本列表
    batch_labels = []  # 初始化批次中的所有标签列表
    for example in batch:  # 遍历批次中的每个样本
        batch_tokens.append(example[0])  # 添加文本
        batch_labels.append(example[1])  # 添加标签
    try:
        # 使用分词器对文本进行编码，包括截断、填充等操作，并返回PyTorch张量
        tokenized_data = tokenizer(batch_tokens, truncation=True, padding=True, return_tensors="pt",
                                   is_split_into_words=True)
    except Exception as e:
        # 如果编码过程中出现异常，打印文本并抛出异常
        print(batch_tokens)
        raise 'xxx'
    max_len = tokenized_data['input_ids'].shape[1]  # 获取编码后的最大长度
    # 创建标签Tensor，每个标签序列前后添加0（对应"O"标签），并确保长度与输入一致
    tokenized_data['labels'] = torch.tensor([([0] + label + [0] * max_len)[:max_len] for label in batch_labels],
                                            dtype=torch.long)
    return tokenized_data  # 返回编码后的数据和标签


# 定义BERT命名实体识别数据集类
class BERTForNERDataset(Dataset):
    # 定义标签到索引的映射
    labels_2_idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}

    def __init__(self, data_path: str, split: str = 'train'):
        super(BERTForNERDataset, self).__init__()
        # 加载数据集，并根据split参数选择训练集、验证集或测试集
        self.data = load_dataset(data_path)[split]
        # 过滤掉长度超过特定阈值的样本
        self.data = self.data.filter(lambda x: 0 < len(x['tokens']) <= 384 - 2)
        # 为每个样本添加长度信息
        self.data = self.data.map(lambda x: {'length': len(x['tokens'])})
        # 根据样本长度进行排序，有助于后续的批量处理
        self.data = self.data.sort('length')

    def __len__(self):
        # 返回数据集中的样本数量
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引获取样本的文本和标签
        tokens = self.data[idx]['tokens']
        label = self.data[idx]['ner_tags']
        return tokens, label  # 返回文本和标签


if __name__ == '__main__':
    dataset = BERTForNERDataset(data_path='./peoples_daily_ner', split='test')
    print(len(dataset))
    print(dataset[0])
    print(BERTForNERDataset.labels_2_idx)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    for batch in loader:
        print(batch)
