# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/12/7 12:50
@File   : predict.py
@desc   : 
"""
import numpy as np
import torch
from transformers import AutoTokenizer
from model import BertForQAModel


# 定义一个函数，用于从模型的输出中获取答案
def get_answer(start_logits, end_logits, tokenized_data, context, n_best=5, max_answer_length=30):
    # 获取每个token在原始文本中的偏移量
    data_offset_mapping = tokenized_data['offset_mapping']
    answers = []  # 存储答案及其分数的列表
    # 遍历每个样本
    for idx in range(start_logits.shape[0]):
        # 获取当前样本的开始和结束logits
        start_logit = start_logits[idx]
        end_logit = end_logits[idx]
        offset = data_offset_mapping[idx]  # 获取当前样本的偏移量

        # 获取开始和结束logits最高的n_best个索引
        start_indexes = np.argsort(start_logit)[::-1][:n_best].tolist()
        end_indexes = np.argsort(end_logit)[::-1][:n_best].tolist()

        # 遍历所有可能的开始和结束索引组合
        for start_index in start_indexes:
            for end_index in end_indexes:
                # 检查偏移量是否有效，以及开始和结束索引是否合理
                if offset[start_index] is None or offset[end_index] is None:
                    continue
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue

                # 添加答案及其分数到列表
                answers.append({
                    "text": context[offset[start_index][0]: offset[end_index][1]],
                    "score": start_logit[start_index] + end_logit[end_index]
                })

    # 如果有答案，选择分数最高的答案
    if len(answers) > 0:
        best_answer = max(answers, key=lambda x: x["score"])
        return best_answer["text"]
    else:
        return ''  # 如果没有答案，返回空字符串


# 定义一个函数，用于预测问题的答案
def predict(question, context, model, tokenizer, device):
    # 使用tokenizer处理问题和上下文
    inputs = tokenizer(
        text=question,
        text_pair=context,
        max_length=384,
        truncation='only_second',
        stride=128,
        padding='max_length',
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_tensors='pt',
    )
    model.eval()  # 将模型设置为评估模式
    start_logits = []  # 存储开始logits
    end_logits = []  # 存储结束logits
    # 在不需要计算梯度的情况下运行模型
    with torch.no_grad():
        # 将输入数据移动到指定设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)  # 获取模型输出
        start_logits.append(out.start_logits)
        end_logits.append(out.end_logits)
        # 将logits列表转换为tensor，并移至CPU
        start_logits = torch.cat(start_logits, dim=0).cpu().detach().numpy()
        end_logits = torch.cat(end_logits, dim=0).cpu().detach().numpy()
    # 获取答案
    answer = get_answer(start_logits, end_logits, inputs, context)
    print(answer)  # 打印答案
    return answer  # 返回答案


if __name__ == '__main__':
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
    # 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化模型并加载预训练参数
    bert_qa = BertForQAModel(num_labels=2).to(device)
    bert_qa.load_state_dict(torch.load("./bert_qa_model.pth"))
    # 上下文
    context = '''春节是中国最重要的传统节日之一，通常在农历正月初一庆祝，标志着新年的开始。‌这个节日有着悠久的历史和丰富的文化内涵，起源于古代的农耕社会，最初是为了祈求丰收和祭祀祖先。春节期间，人们会进行各种庆祝活动，包括贴春联、放鞭炮、吃年夜饭、拜年等，这些活动不仅体现了对美好生活的向往，也加强了家庭和社会的联系。

在历史背景方面，春节的日期在汉代之前并不统一，直到汉武帝时期才固定在农历正月初一。经过几千年的演变，春节的习俗在宋代基本定型，并延续至今。现代春节的庆祝活动丰富多彩，不仅限于家庭团聚，还包括舞龙舞狮、庙会等传统活动，这些活动不仅展示了中华民族的文化精髓，也促进了社会的交流和和谐。

春节不仅仅是一个节日，更是一种文化的传承和表达。通过这个节日，人们不仅祈求新年的好运和吉祥，也表达了对亲朋好友的祝福和思念。春节的庆祝活动如年夜饭、拜年、放鞭炮等，都是为了祈求来年的丰收和幸福，体现了人们对美好生活的向往和对传统文化的尊重。'''

    # 调用predict函数预测答案
    predict(question="春节通常在什么时候庆祝？", context=context, model=bert_qa, tokenizer=tokenizer,
            device=device)
