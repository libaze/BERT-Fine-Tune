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


def predict(sentence, model, tokenizer, device, idx_2_label):
    sen = list(sentence)
    inputs = tokenizer(sen, return_tensors='pt', is_split_into_words=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
        predicted = torch.argmax(output.reshape(-1, output.shape[-1]), 1)
    result = [idx_2_label[i] for i in predicted.data.cpu().numpy().tolist()[1:-1]]
    # print('Predicted:', result)
    # for i in range(len(sen)):
    #     print(sen[i], result[i])
    return sen, result


if __name__ == '__main__':
    num_classes = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_2_idx = BERTForNERDataset.labels_2_idx
    idx_2_label = list(labels_2_idx.keys())
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
    model = BertForNER(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('best_ner_model.pth'))
    model.eval()
    sentence = '即便是对阿根廷队的一场球，中国队在开局后也曾连续10次失误，被对手迅速超过，心态的不稳定可见一斑。'
    predict(sentence, model, tokenizer, device, idx_2_label)

'''
(['未', '来', '也', '许', '是', '中', '国', '女', '篮', '的', '，', '但', '现', '在', '则', '要', '看', '俄', '罗', '斯', '队', '的', '。'], [0, 0, 0, 0, 0, 3, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 0, 0])
(['北', '京', '市', '出', '租', '汽', '车', '管', '理', '局', '是', '唯', '一', '有', '权', '批', '准', '出', '租', '车', '运', '营', '权', '的', '单', '位', '。'], [3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
(['即', '便', '是', '对', '阿', '根', '廷', '队', '的', '一', '场', '球', '，', '中', '国', '队', '在', '开', '局', '后', '也', '曾', '连', '续', '1', '0', '次', '失', '误', '，', '被', '对', '手', '迅', '速', '超', '过', '，', '心', '态', '的', '不', '稳', '定', '可', '见', '一', '斑', '。'], [0, 0, 0, 0, 3, 4, 4, 4, 0, 0, 0, 0, 0, 3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
'''