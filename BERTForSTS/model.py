# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/7/16 10:39
@File   : model.py
@desc   : 
"""
import torch.nn as nn
from transformers import BertModel, BertForNextSentencePrediction


class BertForSTS(nn.Module):
    def __init__(self, num_classes: int = 2):
        # 调用父类的初始化方法
        super(BertForSTS, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained('google-bert/bert-base-chinese')
        # 定义Dropout层，用于防止过拟合，丢弃率设为0.1
        self.dropout = nn.Dropout(0.1)
        # 定义线性层，用于将BERT模型的输出转换为分类得分
        # 输入特征维度为768，输出特征维度为num_classes
        self.seq_relationship = nn.Linear(768, num_classes)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # 将输入传递给BERT模型，获取模型的输出
        outputs = self.bert(
            input_ids=input_ids,            # 输入的token ID序列
            token_type_ids=token_type_ids,  # 分段token ID序列
            attention_mask=attention_mask   # 注意力掩码，避免对padding部分计算注意力
        )
        # 获取BERT模型的输出结果
        pooled_output = outputs.pooler_output
        # 将结果传递给Dropout层
        pooled_output = self.dropout(pooled_output)
        # 将Dropout层的结果传递给线性层，得到分类得分
        seq_relationship_scores = self.seq_relationship(pooled_output)
        # 返回分类得分
        return seq_relationship_scores


