# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/12/7 23:28
@File   : web_ui.py
@desc   : 
"""
import gradio as gr
import torch
from transformers import BertTokenizer
from data import BERTForNERDataset
from predict import predict
from model import BertForNER


num_classes = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels_2_idx = BERTForNERDataset.labels_2_idx
idx_2_label = list(labels_2_idx.keys())
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
model = BertForNER(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('best_ner_model.pth'))
model.eval()


def ner_predict(sentence):
    sen, result = predict(sentence=sentence, model=model, tokenizer=tokenizer, device=device, idx_2_label=idx_2_label)
    # return {k: v for k, v in zip(sen, result)}
    return [sen, result]


with gr.Blocks() as demo:
    gr.Markdown('# 命名实体识别')
    sentence = gr.Text(label='文本')
    df = gr.Dataframe(label='结果')
    gr.Interface(fn=ner_predict, inputs=[sentence], outputs=df,
                 examples=[
                     ['即便是对阿根廷队的一场球，中国队在开局后也曾连续10次失误，被对手迅速超过，心态的不稳定可见一斑。'],
                     ['北京市出租汽车管理局是唯一有权批准出租车运营权的单位。'],
                     ['未来也许是中国女篮的，但现在则要看俄罗斯队的。'],
                 ])


if __name__ == '__main__':
    demo.launch()
