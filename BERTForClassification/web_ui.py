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
from predict import predict
from model import BERTForClassification

num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
model = BERTForClassification(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('best_cls_model.pth'))
model.eval()


def sts_predict(sentence):
    labels = ['差评', '好评']
    pred_score, result = predict(sentence=sentence, model=model, tokenizer=tokenizer, device=device)
    class_score_dict = {labels[i]: float(pred_score[i]) for i in range(len(labels))}
    return class_score_dict


with gr.Blocks() as demo:
    gr.Markdown('# 情感分析')
    sentence = gr.Text(label='评论')
    outputs = gr.Label(num_top_classes=2)
    gr.Interface(fn=sts_predict, inputs=[sentence], outputs=outputs,
                 examples=[
                     ['每次肯定晚至少一个小时，下单了最后没送到，真是醉了，饭也给的死少。不建议大家订餐，如果非要吃，最后提前一天订，第二天再吃'],
                     ['速度很快，下雨天师傅辛苦，味道很不错'],
                     ['下午五点五十五订餐成功，七点半之后才吃上饭，酸奶连影子都没看到，味道也就那么回事，蒜蓉粉丝娃娃菜油特别大，烧二冬都是淀粉，糊嘴都，以后绝对不定这一家了'],
                 ])


if __name__ == '__main__':
    demo.launch()
