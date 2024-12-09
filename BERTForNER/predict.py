# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/7/16 19:58
@File   : predict.py
@desc   : 
"""
import torch
from transformers import BertTokenizer
from data import BERTForNERDataset
from model import BertForNER


# 定义预测函数，用于对给定的句子进行命名实体识别
def predict(sentence, model, tokenizer, device, idx_2_label):
    # 将句子转换为字符列表
    sen = list(sentence)
    # 使用tokenizer对字符列表进行编码，生成模型所需的输入
    inputs = tokenizer(sen, return_tensors='pt', is_split_into_words=True)
    # 将输入数据移动到指定设备（GPU或CPU）
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 禁用梯度计算，因为在预测模式下不需要计算梯度
    with torch.no_grad():
        # 将输入数据传递给模型，获取输出
        output = model(**inputs)
        # 改变output形状，并使用argmax获取最高概率的类别索引
        predicted = torch.argmax(output.reshape(-1, output.shape[-1]), 1)
    # 将预测的类别索引转换为类别标签
    result = [idx_2_label[i] for i in predicted.data.cpu().numpy().tolist()[1:-1]]
    # 返回原始字符列表和对应的预测类别标签
    return sen, result


if __name__ == '__main__':
    # 定义类别数量
    num_classes = 7
    # 检测是否有可用的GPU，如果有则使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将类别标签映射到索引
    labels_2_idx = BERTForNERDataset.labels_2_idx
    # 将类别索引映射回类别标签
    idx_2_label = list(labels_2_idx.keys())
    # 加载预训练的BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
    # 实例化BERT模型，并移动到指定设备
    model = BertForNER(num_classes=num_classes).to(device)
    # 加载训练好的模型权重
    model.load_state_dict(torch.load('best_ner_model.pth'))
    # 将模型设置为评估模式
    model.eval()
    # 定义要预测的句子
    sentence = '即便是对阿根廷队的一场球，中国队在开局后也曾连续10次失误，被对手迅速超过，心态的不稳定可见一斑。'
    # 调用预测函数，并打印结果
    print(predict(sentence, model, tokenizer, device, idx_2_label))

