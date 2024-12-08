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


def get_answer(start_logits, end_logits, processed_data, context, n_best=5, max_answer_length=30):
    data_offset_mapping = processed_data['offset_mapping']
    answers = []
    for idx in range(start_logits.shape[0]):
        start_logit = start_logits[idx]
        end_logit = end_logits[idx]
        offset = data_offset_mapping[idx]
        start_indexes = np.argsort(start_logit)[::-1][:n_best].tolist()
        end_indexes = np.argsort(end_logit)[::-1][:n_best].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                if offset[start_index] is None or offset[end_index] is None:
                    continue
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue
                answers.append({
                    "text": context[offset[start_index][0]: offset[end_index][1]],
                    "score": start_logit[start_index] + end_logit[end_index]
                })
    if len(answers) > 0:
        best_answer = max(answers, key=lambda x: x["score"])
        return best_answer["text"]
    else:
        return ''


def predict(question, context, model, tokenizer, device):
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
    model.eval()
    start_logits = []
    end_logits = []
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        start_logits.append(out.start_logits)
        end_logits.append(out.end_logits)
        start_logits = torch.cat(start_logits, dim=0)
        end_logits = torch.cat(end_logits, dim=0)
        start_logits = start_logits.cpu().detach().numpy()
        end_logits = end_logits.cpu().detach().numpy()
    answer = get_answer(start_logits, end_logits, inputs, context)
    print(answer)
    return answer


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_qa = BertForQAModel(num_labels=2).to(device)
    bert_qa.load_state_dict(torch.load("./bert_qa_model.pth"))
    context = '''2024年12月4日，中国申报的“春节——中国人庆祝传统新年的社会实践”在巴拉圭亚松森举行的联合国教科文组织保护非物质文化遗产政府间委员会第19届常会上通过评审，列入联合国教科文组织人类非物质文化遗产代表作名录。

2024年12月5日，黎族传统纺染织绣技艺、羌年和中国木拱桥传统营造技艺从急需保护的非物质文化遗产名录转入人类非物质文化遗产代表作名录。

非物质文化遗产是中华优秀传统文化的重要组成部分，是中华文明绵延传承的生动见证，是我国各族人民宝贵的精神财富。习近平总书记十分关心非物质文化遗产的保护传承，多次强调“要扎实做好非物质文化遗产的系统性保护”“培养好传承人”等。大象新闻为您梳理，邀您一起学习、领悟、践行。'''
    predict(question="2024年12月4日，中国申报什么通过了评审？", context=context, model=bert_qa, tokenizer=tokenizer, device=device)
