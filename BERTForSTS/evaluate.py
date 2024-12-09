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
    # 将模型设置为评估模式
    model.eval()
    all_acc = 0.  # 初始化正确预测的总数
    # 禁止梯度计算，节省内存和计算资源
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='[Evaluate] '):
            # 将输入数据移动到指定的设备上
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)  # 将标签移动到指定的设备上
            outputs = model(**inputs)  # 前向传播得到模型输出
            # 计算当前批次中正确预测的数量，并累加到总数中
            all_acc += torch.eq(torch.argmax(outputs, dim=1), labels).sum()
    # 计算总的准确率
    return all_acc / len(test_loader.dataset)  # 注意这里应该除以数据集的总数而不是批次的数量


if __name__ == '__main__':
    batch_size = 64  # 批次大小
    num_classes = 2  # 类别数量
    select_num = 20000
    data_path = './data/train_pair.json'  # 数据路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择
    # 创建测试数据集
    test_dataset = BERTForSTSDataset(data_path, 'test', select=select_num)
    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    # 实例化模型并将其移动到指定设备
    model = BertForSTS(num_classes=num_classes).to(device)
    # 加载训练好的模型参数
    model.load_state_dict(torch.load('./best_sts_model.pth'))
    # 调用评估函数并打印测试准确率
    acc = evaluate(model, test_loader, device)
    print('Test Accuracy: {:.2f}%'.format(acc * 100))
    print('Finished Testing!')


