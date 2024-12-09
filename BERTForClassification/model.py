# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/11/23 14:21
@File   : model.py
@desc   : 
"""
from transformers import BertModel, BertForSequenceClassification
from torch import nn


class BERTForClassification(nn.Module):
    def __init__(self, num_classes=2):
        # 调用父类构造函数
        super(BERTForClassification, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained('google-bert/bert-base-chinese')
        # 添加一个Dropout层，用于减少过拟合
        self.dropout = nn.Dropout(0.1)
        # 添加一个线性层作为分类器，输入特征维度为768（BERT模型的输出维度），输出类别数量为num_classes
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # 将输入传递给BERT模型
        outputs = self.bert(
            input_ids=input_ids,            # 输入的token序列
            attention_mask=attention_mask,  # 指示哪些token是实际token，哪些是padding
            token_type_ids=token_type_ids   # 区分句子片段的token类型
        )
        # 获取BERT模型的pooler_output，通常用于分类任务
        pooled_output = outputs.pooler_output
        # 应用Dropout层
        pooled_output = self.dropout(pooled_output)
        # 将dropout后的输出传递给分类器，得到out
        out = self.classifier(pooled_output)
        # out，用于后续的损失计算和分类预测
        return out


