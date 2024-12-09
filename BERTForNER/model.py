# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/7/16 10:39
@File   : model.py
@desc   : 
"""
import torch.nn as nn
from transformers import BertModel, BertForTokenClassification


# 定义一个用于命名实体识别的BERT模型类
class BertForNER(nn.Module):
    def __init__(self, num_classes: int = 7):
        # 调用父类构造函数
        super(BertForNER, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained('google-bert/bert-base-chinese')
        # 定义一个Dropout层，用于减少过拟合
        self.dropout = nn.Dropout(0.1)
        # 定义一个线性层，将BERT模型的输出转换为类别预测
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # 将输入传递给BERT模型
        outputs = self.bert(
            input_ids=input_ids,            # 输入的token ID序列
            token_type_ids=token_type_ids,  # token类型ID序列，用于区分句子片段
            attention_mask=attention_mask   # 注意力掩码，用于避免对padding部分计算注意力
        )
        # 获取BERT模型的序列输出
        sequence_output = outputs.last_hidden_state
        # 应用Dropout层
        sequence_output = self.dropout(sequence_output)
        # 将序列输出传递给线性层进行分类
        out = self.classifier(sequence_output)
        # 返回分类结果
        return out

