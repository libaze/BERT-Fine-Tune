# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/11/8 18:06
@File   : model.py
@desc   : 
"""
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertForQuestionAnswering, AutoTokenizer
from transformers.modeling_outputs import QuestionAnsweringModelOutput


tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')


class BertForQAModel(nn.Module):
    def __init__(self, num_labels):
        super(BertForQAModel, self).__init__()
        self.bert = BertModel.from_pretrained('google-bert/bert-base-chinese')
        self.qa_outputs = nn.Linear(in_features=self.bert.config.hidden_size, out_features=num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)  # 将 logits 张量沿着最后一个维度分割，并且每个分割的子张量的大小为1
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果我们使用多GPU，则split添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时start/end位置超出了我们的模型输入，我们忽略了这些项
            ignored_index = start_logits.size(1)    # d_model大小
            start_positions = start_positions.clamp(0, ignored_index)   # clamp 函数用于将张量中的元素限制在一个特定的范围内
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)  # 计算损失时，忽略超出范围的值
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_qa = BertForQAModel(num_labels=2)
    print(bert_qa)



