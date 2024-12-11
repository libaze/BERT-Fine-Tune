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


# 定义训练模型的函数
def train_model(model, train_dataloader, vaild_dataset, valid_dataloader, optimizer, scheduler):
    best_f1 = 0.  # 初始化最佳F1分数
    epochs = 5  # 训练的轮数
    # 开始训练过程
    for epoch in range(epochs):
        model.train()  # 将模型设置为训练模式
        total_loss = 0  # 初始化总损失
        # 遍历训练数据加载器中的每个批次
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()  # 清除之前的梯度
            out = model(**batch)  # 前向传播
            loss = out.loss  # 获取损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            scheduler.step()  # 更新学习率
            total_loss += loss.item()  # 累加损失
        # 打印当前轮次的平均损失
        print("Epoch: {}/{}, Loss: {:.4f}".format(epoch+1, epochs, total_loss / len(train_dataloader)))
        # 在验证集上评估模型
        avg, f1, em = evaluate_model(model, vaild_dataset, valid_dataloader)
        print("AVG: {:.4f}, F1: {:.4f}, EM: {:.4f}".format(avg, f1, em))  # 打印评估指标
        # 打印当前学习率
        print('current learning rate: ', scheduler.get_last_lr())
        # 如果F1分数有所提高，则保存模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), './bert_qa_model.pth')  # 保存模型参数
            print("Best model saved!")  # 提示模型已保存


if __name__ == '__main__':
    epochs = 5  # 训练轮数
    batch_size = 16  # 批次大小
    learning_rate = 1e-4  # 学习率
    # 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 实例化模型并移动到指定设备
    model = BertForQAModel(num_labels=2).to(device)
    # 加载数据集
    train_dataset = BertForQADataset('./data/cmrc2018', split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    vaild_dataset = BertForQADataset('./data/cmrc2018', split='validation')
    valid_dataloader = DataLoader(vaild_dataset, batch_size=batch_size*2, collate_fn=collate_fn)
    # 实例化优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    T_max, eta_min = epochs * len(train_dataloader), 1e-7  # 设置余弦退火调度器的参数
    scheduler = CosineAnnealingLR(optimizer, T_max, eta_min)
    # 调用train_model函数开始训练模型
    train_model(model, train_dataloader, vaild_dataset, valid_dataloader, optimizer, scheduler)

