# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/11/23 13:53
@File   : train.py
@desc   : 
"""
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate
from model import BERTForClassification
from data.dataset import BertForClassificationDataset, collate_fn


def train(model, train_loader, valid_loader, optimizer, scheduler, loss_fn, epochs, device):
    model.train()
    train_loss = 0
    best_acc = 0.
    for epoch in range(epochs):
        for inputs, labels in tqdm(train_loader):
            # print('lr: ', optimizer.param_groups[0]['lr'])
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            logits = model(**inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        print('Epoch: {}/{}, Loss: {:.4f}'.format(epoch+1, epochs, train_loss / len(train_loader)))
        acc = evaluate(model, valid_loader, device)
        print('Validation Accuracy: {:.4f}'.format(acc))
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_cls_model.pth')
            print('Best Model Saved!')


if __name__ == '__main__':
    epochs = 5
    batch_size = 32
    lr = 5e-5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = BERTForClassification(num_classes=2).to(device)

    train_dataset = BertForClassificationDataset('./data/waimai_10k', 'train')
    val_dataset = BertForClassificationDataset('./data/waimai_10k', 'validation')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, collate_fn=collate_fn)

    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    T_max, eta_min = epochs*len(train_dataset), 1e-7
    scheduler = CosineAnnealingLR(optimizer, T_max, eta_min)

    train(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, epochs, device)
    # acc = evaluate(model, val_dataloader, device)





