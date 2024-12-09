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
        super(BertForSTS, self).__init__()
        self.bert = BertModel.from_pretrained('google-bert/bert-base-chinese')
        self.dropout = nn.Dropout(0.1)
        self.seq_relationship = nn.Linear(768, num_classes)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        seq_relationship_scores = self.seq_relationship(pooled_output)
        return seq_relationship_scores


# if __name__ == '__main__':
#     upper_bound = 100
#     input_ids = torch.randint(0, upper_bound, (4, 5), dtype=torch.long)
#     token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
#     attention_mask = torch.ones_like(input_ids, dtype=torch.long)
#     model = BertForNER(num_classes=upper_bound)
    # out = model(input_ids, token_type_ids, attention_mask)
    # print(out.shape)
    # from transformers import AutoModel, RobertaModel
    # model = RobertaModel.from_pretrained(r'FacebookAI/roberta-base')
    # c = 0
    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param.size())
    #     print(param)
    #     c += 1
    #     if c == 10:
    #         break

