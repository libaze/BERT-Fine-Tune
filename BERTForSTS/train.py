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


# 定义训练函数
def train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device, epochs):
    best_acc = 0.  # 初始化最佳准确率
    for epoch in range(epochs):  # 遍历每个epoch
        model.train()  # 将模型设置为训练模式
        train_loss = 0.  # 初始化训练损失
        all_acc = 0.  # 初始化训练准确率计数
        for inputs, labels in tqdm(train_loader, desc='Epoch: {}/{} '.format(epoch + 1, epochs)):  # 遍历训练数据加载器
            inputs = {k: v.to(device) for k, v in inputs.items()}  # 将输入数据移动到指定设备
            labels = labels.to(device)  # 将标签移动到指定设备
            optimizer.zero_grad()  # 清空梯度
            outputs = model(**inputs)  # 通过模型得到输出
            loss = loss_fn(outputs, labels)  # 计算损失
            train_loss += loss.item()  # 累加损失
            all_acc += torch.eq(torch.argmax(outputs, dim=1), labels).sum()  # 计算准确率
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            scheduler.step()  # 更新学习率
        # 打印训练信息
        print('[Train] Epoch: {}/{}, Loss: {:.4f}, Accuracy: {:.2f} %'.format(
            epoch + 1, epochs, train_loss / len(train_loader), all_acc / len(train_loader.dataset) * 100))
        acc = evaluate(model, valid_loader, device)  # 在验证集上评估模型
        print('[Evaluate] Epoch: {}/{}, Accuracy: {:.2f} %'.format(epoch + 1, epochs, acc * 100))
        # 如果当前准确率比最佳准确率高，则保存模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), './best_sts_model.pth')  # 保存模型参数
            print('Best model saved!')


if __name__ == '__main__':
    batch_size = 64  # 批次大小
    epochs = 5  # 训练轮数
    num_classes = 2  # 分类数量
    lr = 1e-4  # 初始学习率
    select_num = 20000  # 数据集大小（选择数量）
    data_path = './data/train_pair_1w.json'  # 数据文件路径
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备选择
    # 加载数据集
    train_dataset = BERTForSTSDataset(data_path, 'train', select=select_num)
    valid_dataset = BERTForSTSDataset(data_path, 'validation', select=select_num)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
    # 实例化模型
    model = BertForSTS(num_classes=num_classes).to(device)
    # 定义损失函数、优化器和学习率调度器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    T_max, eta_min = epochs*len(train_dataset), 1e-7  # 余弦退火调度器的参数
    scheduler = CosineAnnealingLR(optimizer, T_max, eta_min)
    # 调用train函数开始训练模型
    train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device, epochs)
    print('Finished Training!')  # 训练完成
