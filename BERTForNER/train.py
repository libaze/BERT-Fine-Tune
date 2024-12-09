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
from data import collate_fn, BERTForNERDataset
from model import BertForNER
from evaluate import evaluate, reshape_and_remove_padding, get_accuracy


# 定义训练函数
def train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device, epochs):
    best_acc = 0.  # 初始化最佳准确率
    for epoch in range(epochs):  # 遍历每个epoch
        model.train()  # 将模型设置为训练模式
        train_loss = 0.  # 初始化训练损失
        all_acc = 0.  # 初始化包括'O'标签的总准确率
        all_t_acc = 0.  # 初始化不包括'O'标签的总准确率
        for batch in tqdm(train_loader, desc='Epoch: {}/{} '.format(epoch + 1, epochs)):  # 遍历训练数据加载器
            batch = {k: v.to(device) for k, v in batch.items()}  # 将数据移动到指定设备
            optimizer.zero_grad()  # 清空梯度
            # 前向传播
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])
            outputs, labels = reshape_and_remove_padding(outputs, batch['labels'], batch['attention_mask'])  # 改变形状并移除填充
            correct, total, t_correct, t_total = get_accuracy(outputs, labels)  # 计算准确率
            loss = loss_fn(outputs, labels)  # 计算损失
            train_loss += loss.item()  # 累加损失
            all_acc += correct / total  # 累加包括'O'标签的总准确率
            all_t_acc += t_correct / t_total  # 累加不包括'O'标签的总准确率
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            scheduler.step()  # 更新学习率
        # 打印训练结果
        print('[Train] Epoch: {}/{}, Loss: {:.4f}, Accuracy: {:.2f} %, T-Accuracy: {:.2f} %'.format(epoch + 1, epochs, train_loss / len(train_loader), all_acc / len(train_loader) * 100, all_t_acc / len(train_loader) * 100))
        acc, t_acc = evaluate(model, valid_loader, device)  # 在验证集上评估模型
        # 打印验证结果
        print('[Evaluate] Epoch: {}/{}, Accuracy: {:.2f} %, T-Accuracy: {:.2f} %'.format(epoch + 1, epochs, acc * 100, t_acc * 100))
        # 如果当前不包括'O'标签的总准确率优于最佳准确率，则保存模型
        if t_acc > best_acc:
            best_acc = t_acc
            torch.save(model.state_dict(), './best_ner_model.pth')
            print('Best model saved!')


if __name__ == '__main__':
    batch_size = 64  # 批处理大小
    epochs = 5  # 训练轮数
    num_classes = 7  # 类别数量
    lr = 1e-4  # 学习率
    data_path = './data/peoples_daily_ner'  # 数据路径
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备
    # 加载数据集
    train_dataset = BERTForNERDataset(data_path, 'train')
    valid_dataset = BERTForNERDataset(data_path, 'validation')
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
    # 实例化模型并移动到设备
    model = BertForNER(num_classes=num_classes).to(device)
    # 定义损失函数、优化器和调度器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    T_max, eta_min = epochs*len(train_dataset), 1e-7  # 调度器参数
    scheduler = CosineAnnealingLR(optimizer, T_max, eta_min)
    # 开始训练
    train(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device, epochs)
    print('Finished Training!')  # 训练完成

