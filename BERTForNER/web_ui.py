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


# 定义类别数量，类别数量为7
num_classes = 7
# 检测是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载命名实体识别数据集的标签到索引的映射
labels_2_idx = BERTForNERDataset.labels_2_idx
# 将索引到标签的映射转换为列表形式
idx_2_label = list(labels_2_idx.keys())
# 加载预训练的BERT分词器
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
# 实例化命名实体识别模型，并设置类别数量，然后将模型移动到指定设备
model = BertForNER(num_classes=num_classes).to(device)
# 加载预训练的模型权重
model.load_state_dict(torch.load('best_ner_model.pth'))
# 将模型设置为评估模式
model.eval()


def ner_predict(sentence):
    # 调用predict函数获取预测结果，包括处理后的句子和预测的实体标签
    sen, result = predict(sentence=sentence, model=model, tokenizer=tokenizer, device=device, idx_2_label=idx_2_label)
    # 返回处理后的句子和预测的实体标签列表
    return [sen, result]


with gr.Blocks() as demo:
    gr.Markdown('# 命名实体识别')
    # 添加输入框，用于用户输入文本
    sentence = gr.Text(label='文本')
    # 添加数据框，用于显示预测结果
    df = gr.Dataframe(label='结果')
    # 创建接口，将输入输出与ner_predict函数关联起来，并提供示例文本
    gr.Interface(fn=ner_predict, inputs=[sentence], outputs=df,
                 examples=[
                     ['即便是对阿根廷队的一场球，中国队在开局后也曾连续10次失误，被对手迅速超过，心态的不稳定可见一斑。'],
                     ['北京市出租汽车管理局是唯一有权批准出租车运营权的单位。'],
                     ['郓城县600多名下岗职工，在亚世集团得到了妥善安置，吴佩民心里平静了许多。'],
                 ])


if __name__ == '__main__':
    demo.launch()
