# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/7/16 11:36
@File   : test.py
@desc   : 
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import BERTForNERDataset, collate_fn
from model import BertForNER


# 改变输出和标签，并移除填充部分
def reshape_and_remove_padding(outputs, labels, attention_mask):
    # 输出和标签变换形状
    outputs = outputs.reshape(-1, outputs.shape[-1])
    labels = labels.reshape(-1)
    # 生成一个布尔掩码，用于选择非填充部分
    select = attention_mask.reshape(-1) == 1
    # 使用布尔掩码选择非填充的输出和标签
    outputs = outputs[select]
    labels = labels[select]
    return outputs, labels  # 返回处理后的输出和标签


def get_accuracy(outputs, labels):
    # 计算包括'O'标签在内的准确率
    predicted = torch.argmax(outputs, 1)  # 获取预测的标签索引
    correct = predicted.eq(labels).sum().item()  # 计算正确预测的数量
    total = labels.size(0)  # 获取总的标签数量

    # 计算不包括'O'标签的准确率
    select = labels != 0  # 生成一个布尔掩码，选择非'O'标签
    outputs = outputs[select]  # 使用布尔掩码选择非'O'标签的输出
    labels = labels[select]  # 使用布尔掩码选择非'O'标签的标签
    predicted = torch.argmax(outputs, 1)  # 获取预测的标签索引
    t_correct = predicted.eq(labels).sum().item()  # 计算正确预测的数量
    t_total = labels.size(0)  # 获取非'O'标签的总数量
    return correct, total, t_correct, t_total  # 返回包括'O'和不包括'O'的准确率


def evaluate(model, test_loader, device):
    model.eval()  # 将模型设置为评估模式
    all_acc = 0.  # 初始化包括'O'标签的总准确率
    all_t_acc = 0.  # 初始化不包括'O'标签的总准确率
    with torch.no_grad():  # 确保在评估过程中不会进行梯度计算
        for batch in tqdm(test_loader, desc='[Evaluate] '):  # 遍历测试数据加载器
            batch = {k: v.to(device) for k, v in batch.items()}  # 将数据移动到指定设备
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])  # 获取模型输出
            outputs, labels = reshape_and_remove_padding(outputs, batch['labels'], batch['attention_mask'])  # 处理输出和标签
            correct, total, t_correct, t_total = get_accuracy(outputs, labels)  # 计算准确率
            all_acc += correct / total  # 累加包括'O'标签的准确率
            all_t_acc += t_correct / t_total  # 累加不包括'O'标签的准确率
    return all_acc / len(test_loader), all_t_acc / len(test_loader)  # 返回平均准确率


if __name__ == '__main__':
    # 设置批处理大小和类别数量
    batch_size = 64
    num_classes = 7
    # 设置数据路径
    data_path = './data/peoples_daily_ner'
    # 检测并指定使用GPU还是CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载测试数据集
    test_dataset = BERTForNERDataset(data_path, 'test')
    # 创建测试数据加载器，使用指定的批处理大小和自定义的collate_fn函数
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    # 实例化BERT模型，用于命名实体识别，并指定类别数量
    model = BertForNER(num_classes=num_classes).to(device)
    # 加载训练好的模型权重
    model.load_state_dict(torch.load('./best_ner_model.pth'))
    # 在测试集上评估模型性能，并获取包括和不包括'O'标签的准确率
    acc, t_acc = evaluate(model, test_loader, device)
    # 打印测试准确率，包括所有标签和排除'O'标签的准确率
    print('Test Accuracy: {:.2f}%, T-Accuracy: {:.2f}%'.format(acc * 100, t_acc * 100))
    # 打印完成测试的消息
    print('Finished Testing!')


