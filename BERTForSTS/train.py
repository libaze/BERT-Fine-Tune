# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/7/16 10:53
@File   : train.py
@desc   : 
"""
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import collate_fn, BERTForSTSDataset
from model import BertForSTS
from evaluate import evaluate


def train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device, epochs):
    best_acc = 0.
    for epoch in range(epochs):
        model.train()
        train_loss = 0.
        all_acc = 0.
        for inputs, labels in tqdm(train_loader, desc='Epoch: {}/{} '.format(epoch + 1, epochs)):
            # print('current lr: ', optimizer.param_groups[0]['lr'])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = loss_fn(outputs, labels)
            train_loss += loss.item()
            all_acc += torch.eq(torch.argmax(outputs, dim=1), labels).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print('[Train] Epoch: {}/{}, Loss: {:.4f}, Accuracy: {:.2f} %'.format(epoch + 1, epochs, train_loss / len(train_loader), all_acc / len(train_loader) * 100))
        acc = evaluate(model, valid_loader, device)
        print('[Evaluate] Epoch: {}/{}, Accuracy: {:.2f} %'.format(epoch + 1, epochs, acc * 100))

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), './best_sts_model.pth')
            print('Best model saved!')


if __name__ == '__main__':
    batch_size = 64
    epochs = 10
    num_classes = 2
    lr = 1e-4
    data_path = './data/train_pair_1w.json'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = BERTForSTSDataset(data_path, 'train')
    valid_dataset = BERTForSTSDataset(data_path, 'validation')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)

    model = BertForSTS(num_classes=num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    T_max, eta_min = epochs*len(train_dataset), 1e-7
    scheduler = CosineAnnealingLR(optimizer, T_max, eta_min)

    train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device, epochs)

    print('Finished Training!')

