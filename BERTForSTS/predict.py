# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/7/16 19:58
@File   : predict.py
@desc   : 
"""
import numpy as np
import torch
from transformers import BertTokenizer
from model import BertForSTS


def predict(s1, s2, model, tokenizer, device):
    inputs = tokenizer(
        text=s1,
        text_pair=s2,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
        output = torch.softmax(output, dim=1)
        result = torch.argmax(output)
    return output.cpu().numpy().tolist()[0], result


if __name__ == '__main__':
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
    model = BertForSTS(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('best_sts_model.pth'))
    model.eval()
    predict(s1='我？弗丽达问，眼睛仍旧没有离开窗外。', s2='怎么！这个女人还跟他说你呢！', model=model, tokenizer=tokenizer, device=device)


    # while True:
    #     sen = input('Input Text:')

        # with torch.no_grad():
        #     output = model(**inputs)
        #     predicted = torch.argmax(output.reshape(-1, output.shape[-1]), 1)
        # result = [idx_2_label[i] for i in predicted.data.cpu().numpy().tolist()[1:-1]]
        # print('Predicted:', result)
        # print(len(sen) == len(result))
        # for i in range(len(sen)):
        #     print(sen[i], result[i])
        # print('Finished Predicting!')
