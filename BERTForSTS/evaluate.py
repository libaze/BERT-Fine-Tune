# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/7/16 11:36
@File   : test.py
@desc   : 
"""
import torch
from tqdm import tqdm
from model import BertForSTS
from torch.utils.data import DataLoader
from data import BERTForSTSDataset, collate_fn


def evaluate(model, test_loader, device):
    model.eval()
    all_acc = 0.
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='[Evaluate] '):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            outputs = model(**inputs)
            all_acc += torch.eq(torch.argmax(outputs, dim=1), labels).sum()
    return all_acc / len(test_loader)


if __name__ == '__main__':
    batch_size = 64
    num_classes = 2
    data_path = './data/train_pair.json'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = BERTForSTSDataset(data_path, 'test', select=20000)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    model = BertForSTS(num_classes=num_classes).to(device)

    model.load_state_dict(torch.load('./best_sts_model.pth'))

    acc = evaluate(model, test_loader, device)
    print('Test Accuracy: {:.2f}%'.format(acc * 100))
    print('Finished Testing!')


