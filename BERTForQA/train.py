# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/11/9 13:03
@File   : train.py
@desc   : 
"""
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from data.dataset import BertForQADataset, collate_fn
from evaluate import evaluate_model
from model import BertForQAModel
from torch.utils.data import DataLoader


def train_model(model, train_dataloader, vaild_dataset, valid_dataloader, optimizer, scheduler):
    best_f1 = 0.
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Epoch: {}/{}, Loss: {:.4f}".format(epoch+1, epochs, total_loss / len(train_dataloader)))
        avg, f1, em = evaluate_model(model, vaild_dataset, valid_dataloader)
        print("AVG: {:.4f}, F1: {:.4f}, EM: {:.4f}".format(avg, f1, em))
        scheduler.step()
        print('current learning rate: ', scheduler.get_last_lr())
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), './bert_qa_model1.pth')
            print("Best model saved!")


if __name__ == '__main__':
    epochs = 5
    batch_size = 16
    learning_rate = 5e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T_max, eta_min = epochs, 0
    model = BertForQAModel(num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max, eta_min)
    train_dataset = BertForQADataset('./data/cmrc2018', split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    vaild_dataset = BertForQADataset('./data/cmrc2018', split='validation')
    valid_dataloader = DataLoader(vaild_dataset, batch_size=batch_size, collate_fn=collate_fn)

    train_model(model, train_dataloader, vaild_dataset, valid_dataloader, optimizer, scheduler)





