# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/11/23 19:58
@File   : predict.py
@desc   : 
"""
import torch
from transformers import BertTokenizer
from model import BERTForClassification


def predict(sentence, model, tokenizer, device):
    # 使用tokenizer对输入句子进行编码
    # max_length指定最大长度，truncation进行截断，padding进行填充
    # return_tensors="pt"表示返回PyTorch张量
    inputs = tokenizer(
        sentence,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )
    # 将编码后的输入移动到指定的设备（如GPU或CPU）
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 禁用梯度计算，因为我们是在预测模式下
    with torch.no_grad():
        # 将编码后的输入传递给模型
        output = model(**inputs)
        # 应用softmax函数将输出转换为概率分布
        output = torch.softmax(output, dim=1)
        # 获取概率最高的类别索引
        result = torch.argmax(output)
    # 将输出和结果移动回CPU，并转换为列表形式返回
    return output.cpu().numpy().tolist()[0], result


if __name__ == '__main__':
    # 定义模型的类别数量，二分类问题
    num_classes = 2
    # 指定设备，如果CUDA可用，则使用第一个GPU，否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载预训练的BERT分词器，这里使用的是谷歌提供的中文BERT模型
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
    # 实例化BERT分类模型，并指定类别数量
    model = BERTForClassification(num_classes=num_classes)
    # 将模型移动到指定的设备上，以便进行后续的预测操作
    model.to(device)
    # 加载模型权重，这里假设已经有一个训练好的模型权重文件'best_cls_model.pth'
    model.load_state_dict(torch.load('best_cls_model.pth'))
    # 将模型设置为评估模式，关闭Dropout和BatchNorm层，提高预测的稳定性
    model.eval()
    # 使用模型进行预测，传入一个句子，模型，分词器以及设备信息
    # 打印出预测结果，包括概率分布和预测类别
    print(predict(sentence='特别快,感觉定完没几分钟就来了,送餐小哥很细心,赞一个',
                  model=model, tokenizer=tokenizer, device=device))
