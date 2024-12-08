# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/7/16 11:36
@File   : test.py
@desc   : 
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from data import BERTForNERDataset, collate_fn
from model import BertForNER


def reshape_and_remove_padding(outputs, labels, attention_mask):
    outputs = outputs.reshape(-1, outputs.shape[-1])
    labels = labels.reshape(-1)

    select = attention_mask.reshape(-1) == 1
    outputs = outputs[select]
    labels = labels[select]
    return outputs, labels


def get_accuracy(outputs, labels):
    # 包括'O'
    predicted = torch.argmax(outputs, 1)
    correct = predicted.eq(labels).sum().item()
    total = labels.size(0)
    # 不包括'O'
    select = labels != 0
    outputs = outputs[select]
    labels = labels[select]
    predicted = torch.argmax(outputs, 1)
    t_correct = predicted.eq(labels).sum().item()
    t_total = labels.size(0)
    return correct, total, t_correct, t_total


def evaluate(model, test_loader, device):
    model.eval()
    all_acc = 0.
    all_t_acc = 0.
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='[Evaluate] '):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])
            outputs, labels = reshape_and_remove_padding(outputs, batch['labels'], batch['attention_mask'])
            correct, total, t_correct, t_total = get_accuracy(outputs, labels)
            all_acc += correct / total
            all_t_acc += t_correct / t_total
    return all_acc / len(test_loader), all_t_acc / len(test_loader)


if __name__ == '__main__':
    batch_size = 64
    num_classes = 7
    data_path = './data/peoples_daily_ner'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = BERTForNERDataset(data_path, 'test')

    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    model = BertForNER(num_classes=num_classes).to(device)

    model.load_state_dict(torch.load('./best_ner_model.pth'))

    acc, t_acc = evaluate(model, test_loader, device)
    print('Test Accuracy: {:.2f}%, T-Accuracy: {:.2f}%'.format(acc * 100, t_acc * 100))
    print('Finished Testing!')


