# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/11/23 22:10
@File   : evaluate.py
@desc   : 
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import BERTForClassification
from data.dataset import BertForClassificationDataset, collate_fn


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            logits = model(**inputs)
            preds = logits.argmax(dim=-1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


if __name__ == '__main__':
    batch_size = 32

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = BERTForClassification(num_classes=2)
    model.load_state_dict(torch.load("./best_cls_model.pth"))
    model.to(device)
    test_dataset = BertForClassificationDataset('./data/waimai_10k', 'test')

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn)

    acc = evaluate(model, test_dataloader, device)
    print('Test Accuracy: {:.4f}'.format(acc))
