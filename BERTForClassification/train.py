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
    # 将模型设置为训练模式
    model.train()
    train_loss = 0  # 初始化训练损失
    best_acc = 0.  # 初始化最佳准确率
    # 遍历每一个训练轮次
    for epoch in range(epochs):
        # 使用tqdm显示进度条
        for inputs, labels in tqdm(train_loader):
            # 清空优化器的梯度
            optimizer.zero_grad()
            # 将输入数据移动到指定设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            # 前向传播，获取模型输出
            logits = model(**inputs)
            # 计算损失
            loss = loss_fn(logits, labels)
            # 反向传播，计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 更新学习率
            scheduler.step()
            # 累加训练损失
            train_loss += loss.item()
        # 打印当前轮次的平均损失
        print('Epoch: {}/{}, Loss: {:.4f}'.format(epoch + 1, epochs, train_loss / len(train_loader)))
        # 在验证集上评估模型性能
        acc = evaluate(model, valid_loader, device)
        # 打印验证集上的准确率
        print('Validation Accuracy: {:.4f}'.format(acc))
        # 如果当前准确率优于最佳准确率，则保存模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_cls_model.pth')
            print('Best Model Saved!')


if __name__ == '__main__':
    # 训练参数设置
    epochs = 5  # 训练的总轮数
    batch_size = 32  # 训练的批量大小
    lr = 5e-5  # 学习率

    # 设备选择，如果有可用的GPU，则使用第一个GPU，否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 实例化BERT分类模型，并指定类别数量，然后将模型移动到指定设备
    model = BERTForClassification(num_classes=2).to(device)

    # 加载数据集，这里假设数据集是以特定格式存储的，并且提供了训练集和验证集
    train_dataset = BertForClassificationDataset('./data/waimai_10k', 'train')
    val_dataset = BertForClassificationDataset('./data/waimai_10k', 'validation')

    # 创建DataLoader，用于批量加载数据集，并指定整理函数collate_fn
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn)

    # 定义损失函数，这里使用交叉熵损失函数，适用于分类问题
    loss_fn = CrossEntropyLoss()

    # 定义优化器，这里使用AdamW优化器，它是Adam优化器的变种，适用于BERT模型
    optimizer = AdamW(model.parameters(), lr=lr)

    # 设置学习率调度器，这里使用余弦退火调度器，它会在每个epoch后调整学习率
    T_max, eta_min = epochs * len(train_dataset), 1e-7  # 设置调度器的周期和最小学习率
    scheduler = CosineAnnealingLR(optimizer, T_max, eta_min)

    # 调用训练函数开始训练模型，传入模型、数据加载器、优化器、调度器、损失函数、训练轮数和设备
    train(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, epochs, device)
