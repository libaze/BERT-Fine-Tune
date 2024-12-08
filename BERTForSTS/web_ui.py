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
from model import BertForSTS

num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
model = BertForSTS(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('best_sts_model.pth'))
model.eval()


def sts_predict(s1, s2):
    labels = ['不相似', '相似']
    pred_score, result = predict(s1=s1, s2=s2, model=model, tokenizer=tokenizer, device=device)
    class_score_dict = {labels[i]: float(pred_score[i]) for i in range(len(labels))}
    return class_score_dict


with gr.Blocks() as demo:
    gr.Markdown('# 语义文本相似度')
    s1 = gr.Text(label='句子1')
    s2 = gr.Text(label='句子2')
    outputs = gr.Label(num_top_classes=2)
    gr.Interface(fn=sts_predict, inputs=[s1, s2], outputs=outputs,
                 examples=[
                     ['一只狗和一个女人坐在户外咖啡馆里。', '"一个男人和一只猫坐在室内。'],
                     ['另外还有肥壮的油光水滑的马，依毛色归类，盖着各种颜色的马衣，用短缰绳系在髙高的架木上，胆怯地往后边斜眼望着它们的马贩子老板手里的、它们十分熟悉的鞭子；草原贵族们从一两百俄里外送出来的家养的马，由一个衰老的马车夫和两三个迟钝的马夫监视着，摇晃着它们的长长的脖子，躲着脚，不耐烦地哨着木桩子；黄褐色的维亚特种马紧紧地互相偎依着，尾巴波浪形、蹄上毛的、臀部宽阔的跑大步的马，有灰色带圆斑的，有乌黑的，有専红色的，都像狮子一般庄严稳健地站着。', '忽然，有一天，请您想像，阿丽娜——她名叫阿丽娜——没有禀告就走进了我的书房，——扑通一声向我跪下了。'],
                     ['然而，真正的答案是，科学不能告诉我们除了合理怀疑之外发生了什么。', '科学不能告诉我们发生了什么，不能排除合理的怀疑。'],
                 ])


if __name__ == '__main__':
    demo.launch()
