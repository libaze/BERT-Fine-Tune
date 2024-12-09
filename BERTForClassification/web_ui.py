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


# 定义类别数量
num_classes = 2
# 检测是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载预训练的BERT分词器
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
# 实例化分类模型，并设置类别数量，然后将模型移动到指定设备
model = BERTForClassification(num_classes=num_classes).to(device)
# 加载预训练的模型权重
model.load_state_dict(torch.load('best_cls_model.pth'))
# 将模型设置为评估模式
model.eval()


def cls_predict(sentence):
    # 定义标签列表，对应差评和好评
    labels = ['差评', '好评']
    # 调用predict函数获取预测分数和结果
    pred_score, result = predict(sentence=sentence, model=model, tokenizer=tokenizer, device=device)
    # 将预测分数和标签对应起来，并转换为字典格式
    class_score_dict = {labels[i]: float(pred_score[i]) for i in range(len(labels))}
    return class_score_dict


with gr.Blocks() as demo:
    gr.Markdown('# 情感分析')
    # 添加输入框，用于用户输入评论
    sentence = gr.Text(label='评论')
    # 添加输出标签，用于显示预测结果
    outputs = gr.Label(num_top_classes=2)
    # 创建接口，将输入输出与cls_predict函数关联起来，并提供示例评论
    gr.Interface(fn=cls_predict, inputs=[sentence], outputs=outputs,
                 examples=[
                     ['每次肯定晚至少一个小时，下单了最后没送到，真是醉了，饭也给的死少。不建议大家订餐，如果非要吃，最后提前一天订，第二天再吃'],
                     ['速度很快，下雨天师傅辛苦，味道很不错'],
                     ['下午五点五十五订餐成功，七点半之后才吃上饭，酸奶连影子都没看到，味道也就那么回事，蒜蓉粉丝娃娃菜油特别大，烧二冬都是淀粉，糊嘴都，以后绝对不定这一家了'],
                 ])


if __name__ == '__main__':
    demo.launch()
