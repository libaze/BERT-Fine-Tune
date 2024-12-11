# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/12/7 17:56
@File   : web_ui.py
@desc   : 
"""
import gradio as gr
import torch
from transformers import AutoTokenizer
from model import BertForQAModel
from predict import predict

# 加载预训练的分词器和模型
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_qa = BertForQAModel(num_labels=2).to(device)
bert_qa.load_state_dict(torch.load("./bert_qa_model.pth"))  # 加载模型参数


# 定义一个函数，用于预测给定问题和上下文中的答案
def qa_predict(question: str, context: str):
    answer = predict(question=question, context=context, model=bert_qa, tokenizer=tokenizer, device=device)
    return answer


with gr.Blocks() as demo:
    gr.Markdown('# 问答任务')
    # 创建输入组件：问题、上下文
    question = gr.Text(label='问题')
    context = gr.Text(label='上下文')
    # 创建输出组件：答案
    answer = gr.Text(label='答案')
    # 创建接口，将输入组件和输出组件关联起来，并提供示例
    gr.Interface(qa_predict, [question, context], [answer], examples=[
        ['联合国教科文组织保护非物质文化遗产政府间委员会第19届常会在哪里举行的？', ''''2024年12月4日，中国申报的“春节——中国人庆祝传统新年的社会实践”在巴拉圭亚松森举行的联合国教科文组织保护非物质文化遗产政府间委员会第19届常会上通过评审，列入联合国教科文组织人类非物质文化遗产代表作名录。
2024年12月5日，黎族传统纺染织绣技艺、羌年和中国木拱桥传统营造技艺从急需保护的非物质文化遗产名录转入人类非物质文化遗产代表作名录。
非物质文化遗产是中华优秀传统文化的重要组成部分，是中华文明绵延传承的生动见证，是我国各族人民宝贵的精神财富。习近平总书记十分关心非物质文化遗产的保护传承，多次强调“要扎实做好非物质文化遗产的系统性保护”“培养好传承人”等。大象新闻为您梳理，邀您一起学习、领悟、践行。'''],
        ['庆祝中华人民共和国成立70周年大会在哪里举行？',
         '''10月1日上午，庆祝中华人民共和国成立70周年大会在北京隆重举行。阅兵仪式结束之后，致敬方阵的21辆礼宾车上，坐着老一辈党和国家、军队领导人亲属代表，老一辈建设者和家属代表，新中国成立前参加革命工作的老战士，老一辈军队退役英模、民兵英模和支前模范代表。感人的场面，让无数人泪奔。这是以高规格的方式向共和国的英雄和先锋致敬，充分体现了以习近平同志为核心的党中央对老干部、对英模的尊重和关怀。'''],
        ['BERT是哪年提出的？',
         '''BERT是一种基于Transformer的预训练语言处理模型。由Google于2018年提出，BERT通过双向训练方式，能同时考虑文本上下文信息，提高了模型对语境的理解能力。预训练过程中，BERT采用掩码语言模型（Masked Language Model, MLM）和下一句预测（Next Sentence Prediction, NSP）任务，学习到丰富的语言表示。BERT在多项自然语言处理任务中取得了显著效果，如文本分类、命名实体识别、情感分析等，成为当前自然语言处理领域的重要技术之一。''']
    ])

if __name__ == '__main__':
    demo.launch()
