# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/11/9 20:35
@File   : evaluate.py
@desc   : 
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.cmrc2018_evaluate import evaluate
from data.dataset import BertForQADataset, get_answer_and_prediction_list, collate_fn
from model import BertForQAModel
import time


# 定义评估模型的函数
def evaluate_model(model, dataset, dataloader):
    # 将模型设置为评估模式
    model.eval()
    start_logits = []  # 用于存储开始位置的logits
    end_logits = []    # 用于存储结束位置的logits
    # 不计算梯度
    with torch.no_grad():
        # 遍历数据加载器中的每个批次
        for batch in tqdm(dataloader):
            # 将批次数据传递给模型，并获取输出
            out = model(**batch)
            # 将输出中的开始和结束logits添加到列表中
            start_logits.append(out.start_logits)
            end_logits.append(out.end_logits)
        # 将所有批次的logits沿着第一个维度（通常是批次维度）拼接起来
        start_logits = torch.cat(start_logits, dim=0)
        end_logits = torch.cat(end_logits, dim=0)
        # 将logits移动到CPU上，并转换为numpy数组
        start_logits = start_logits.cpu().detach().numpy()
        end_logits = end_logits.cpu().detach().numpy()

    # 记录评估开始时间
    start_time = time.time()
    # 获取真实答案和预测答案
    ground_trues, predictions = get_answer_and_prediction_list(start_logits, end_logits, dataset, dataset.data)
    # 计算精确率、召回率和F1分数
    F1, EM, TOTAL, SKIP = evaluate(ground_trues, predictions)
    # 打印评估所需时间
    print('Time: ', time.time() - start_time)
    # 计算平均分数
    AVG = (EM + F1) * 0.5
    # 打印各项评估指标
    print('EM score:', EM)
    print('F1 score:', F1)
    print('TOTAL score:', TOTAL)
    print('SKIP score:', SKIP)
    print('AVG score:', AVG)
    # 返回平均分数、F1分数和EM分数
    return AVG, F1, EM


if __name__ == '__main__':
    # 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 实例化模型
    bert_qa = BertForQAModel(num_labels=2)
    # 加载模型参数
    bert_qa.load_state_dict(torch.load("./bert_qa_model.pth"))
    # 将模型移动到指定设备
    bert_qa.to(device)
    # 加载数据集
    dataset = BertForQADataset('./data/cmrc2018', split='test')
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    # 调用评估模型函数
    evaluate_model(bert_qa, dataset, dataloader)


