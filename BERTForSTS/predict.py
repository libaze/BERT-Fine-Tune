# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/7/16 19:58
@File   : predict.py
@desc   : 
"""
import torch
from transformers import BertTokenizer
from model import BertForSTS


def predict(s1, s2, model, tokenizer, device):
    # 使用分词器对输入的两个句子进行编码，准备输入模型
    inputs = tokenizer(
        text=s1,  # 第一个句子
        text_pair=s2,  # 第二个句子
        max_length=128,  # 设置最大长度为128个token
        truncation=True,  # 如果长度超过最大长度，则进行截断
        padding='max_length',  # 对句子进行填充，使它们达到最大长度
        return_tensors="pt"  # 返回PyTorch张量
    )
    # 将编码后的输入数据移动到指定设备上（CPU或GPU）
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 不计算梯度
    with torch.no_grad():
        # 将处理后的输入数据传递给模型，得到模型的输出
        output = model(**inputs)
        # 使用softmax函数对模型的输出进行归一化，得到概率分布
        output = torch.softmax(output, dim=1)
        # 获取概率最高的类别索引，即预测结果
        result = torch.argmax(output)
    # 将预测的概率分布和结果转换为列表形式，并返回
    return output.cpu().numpy().tolist()[0], result


if __name__ == '__main__':
    # 定义分类的数量
    num_classes = 2
    # 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载预训练的分词器
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
    # 实例化BertForSTS模型，并指定分类数量
    model = BertForSTS(num_classes=num_classes).to(device)
    # 加载已经训练好的模型参数
    model.load_state_dict(torch.load('best_sts_model.pth'))
    # 将模型设置为评估模式
    model.eval()
    # 调用predict函数，传入两个句子、模型、分词器和设备，得到预测结果
    predict(
        s1='我？弗丽达问，眼睛仍旧没有离开窗外。',  # 第一个句子
        s2='怎么！这个女人还跟他说你呢！',  # 第二个句子
        model=model,  # 模型实例
        tokenizer=tokenizer,  # 分词器实例
        device=device  # 设备
    )

