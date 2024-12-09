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
    # 将模型设置为评估模式
    model.eval()
    correct = 0  # 初始化正确预测的数量
    total = 0  # 初始化总的样本数量
    # 在评估过程中不计算梯度
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            # 将输入数据移动到指定设备，并压缩不必要的维度
            inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            # 前向传播，获取模型输出
            out = model(**inputs)
            # 获取预测结果，即概率最高的类别索引
            preds = out.argmax(dim=-1)
            # 累加总的样本数量
            total += labels.size(0)
            # 累加正确预测的数量
            correct += (preds == labels).sum().item()
    # 计算准确率并返回
    return correct / total


if __name__ == '__main__':
    batch_size = 32  # 定义批量大小
    # 检测是否有可用的GPU，如果有则使用第一个GPU，否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 实例化分类模型，并加载预训练的权重
    model = BERTForClassification(num_classes=2)
    model.load_state_dict(torch.load("./best_cls_model.pth"))
    model.to(device)  # 将模型移动到指定设备
    # 加载测试数据集
    test_dataset = BertForClassificationDataset('./data/waimai_10k', 'test')
    # 创建测试数据加载器，不进行打乱，并指定批量大小和整理函数
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn)
    # 在测试集上评估模型性能
    acc = evaluate(model, test_dataloader, device)
    # 打印测试准确率
    print('Test Accuracy: {:.4f}'.format(acc))

