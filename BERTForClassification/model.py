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
        super(BERTForClassification, self).__init__()
        self.bert = BertModel.from_pretrained('google-bert/bert-base-chinese')
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


if __name__ == '__main__':
    bert = BERTForClassification()
