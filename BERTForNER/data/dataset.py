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

tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')


def collate_fn(batch):
    batch_tokens = []
    batch_labels = []
    for example in batch:
        batch_tokens.append(example[0])
        batch_labels.append(example[1])
    try:
        tokenized_data = tokenizer(batch_tokens, truncation=True, padding=True, return_tensors="pt", is_split_into_words=True)
    except:
        print(batch_tokens)
        raise 'xxx'
    max_len = tokenized_data['input_ids'].shape[1]
    tokenized_data['labels'] = torch.tensor([([0] + label + [0]*max_len)[:max_len] for label in batch_labels], dtype=torch.long)
    return tokenized_data


class BERTForNERDataset(Dataset):
    # ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    labels_2_idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}

    def __init__(self, data_path: str, split: str = 'train'):
        super(BERTForNERDataset, self).__init__()
        self.data = load_dataset(data_path)[split]
        self.data = self.data.filter(lambda x: 0 < len(x['tokens']) <= 384-2)
        self.data = self.data.map(lambda x: {'length': len(x['tokens'])})
        self.data = self.data.sort('length')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]['tokens']
        label = self.data[idx]['ner_tags']
        return tokens, label


if __name__ == '__main__':
    dataset = BERTForNERDataset(data_path='./peoples_daily_ner', split='train')
    print(len(dataset))
    print(dataset[0])
    print(BERTForNERDataset.labels_2_idx)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    for batch in loader:
        print(batch)



