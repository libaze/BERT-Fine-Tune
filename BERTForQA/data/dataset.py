# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/11/8 17:06
@File   : dataset.py
@desc   : 
"""
import concurrent
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 从模型的输出中获取预测结果
def get_predict_result(example, ids_to_tokenized_data, data_offset_mapping, start_logits, end_logits, n_best=5,
                       max_answer_length=30):
    # 获取当前示例的ID
    _id = example['id']
    # 获取当前示例的上下文
    context = example["context"]
    # 初始化答案列表
    answers = []

    # 遍历与当前示例ID对应的tokenized数据索引
    for idx in ids_to_tokenized_data[_id]:
        # 获取当前索引的起始和结束logits
        start_logit = start_logits[idx]
        end_logit = end_logits[idx]
        # 获取当前索引的偏移映射
        offset = data_offset_mapping[idx]

        # 获取起始logits的前n_best个最大值索引
        start_indexes = np.argsort(start_logit)[::-1][:n_best].tolist()
        # 获取结束logits的前n_best个最大值索引
        end_indexes = np.argsort(end_logit)[::-1][:n_best].tolist()

        # 遍历起始和结束索引，组合可能的答案
        for start_index in start_indexes:
            for end_index in end_indexes:
                # 如果偏移映射为None，则跳过-->过滤
                if offset[start_index] is None or offset[end_index] is None:
                    continue
                # 如果结束索引小于起始索引，或者答案长度超过最大长度，则跳过-->过滤
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue
                # 添加答案到列表
                answers.append({
                    "text": context[offset[start_index][0]: offset[end_index][1]],
                    "score": start_logit[start_index] + end_logit[end_index]
                })

    # 如果答案列表不为空，则选择得分最高的答案
    if len(answers) > 0:
        best_answer = max(answers, key=lambda x: x["score"])
        return example["answers"]["text"], best_answer["text"]
    else:
        # 如果没有找到答案，则返回空字符串
        return example["answers"]["text"], ''


# 获取所有示例的答案和预测列表
def get_answer_and_prediction_list(start_logits, end_logits, tokenized_data, examples, max_workers=4):
    # 创建一个默认字典，用于存储每个ID对应的tokenized数据索引
    ids_to_tokenized_data = defaultdict(list)
    for idx, item in enumerate(tokenized_data):
        ids_to_tokenized_data[item['data_ids']].append(idx)
    # {'TRIAL_800_QUERY_0': [0], 'TRIAL_800_QUERY_1': [1, 2], ...}
    # 获取偏移映射
    data_offset_mapping = tokenized_data['offset_mapping']

    # 设置参数
    n_best = 5
    max_answer_length = 30

    # 初始化真实答案和预测答案列表
    ground_trues = []
    predictions = []

    # 使用线程池执行器并行处理每个示例的预测结果
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务到线程池
        futures = [
            executor.submit(get_predict_result, example, ids_to_tokenized_data, data_offset_mapping, start_logits,
                            end_logits, n_best, max_answer_length) for example in examples]
        # 等待所有任务完成，并获取结果
        for future in concurrent.futures.as_completed(futures):
            gt, pred = future.result()
            ground_trues.append(gt)
            predictions.append(pred)

    # 返回真实答案和预测答案列表
    return ground_trues, predictions


# 处理问答数据
def data_process(data):
    # 使用分词器对问题和上下文进行编码，设置最大长度，截断策略，步长，填充策略等
    data_tokenized = tokenizer(
        text=data['question'],  # 问题文本
        text_pair=data['context'],  # 上下文文本
        max_length=384,  # 设置序列的最大长度
        truncation='only_second',  # 只对上下文进行截断
        stride=128,  # 当文本被截断时，用于创建重叠的片段
        padding='max_length',  # 填充到最大长度
        return_overflowing_tokens=True,  # 返回溢出的标记
        return_offsets_mapping=True,  # 返回偏移映射
    )

    # 提取数据中的溢出标记到样本的映射
    data_overflow_to_sample_mapping = data_tokenized['overflow_to_sample_mapping']
    # 提取偏移映射，用于将token映射回原始文本
    data_offset_mapping = data_tokenized['offset_mapping']
    # 提取token类型id，用于区分问题和上下文
    data_token_type_ids = data_tokenized['token_type_ids']

    # 初始化答案开始和结束位置的列表
    answer_starts = []
    answer_ends = []
    # 初始化数据ID列表
    data_ids = []

    for idx, _ in enumerate(data_overflow_to_sample_mapping):
        # 获取当前样本的答案开始和结束位置
        answer_start = data['answers'][data_overflow_to_sample_mapping[idx]]['answer_start'][0]
        answer_end = answer_start + len(data['answers'][data_overflow_to_sample_mapping[idx]]['text'][0])

        # 获取上下文在token序列中的开始和结束位置-->边界
        context_start = data_token_type_ids[idx].index(1)  # 上下文开始的位置
        context_end = data_tokenized.sequence_ids(idx)[context_start:].index(None) - 1  # 上下文结束的位置

        # 获取当前样本的偏移映射
        offset = data_offset_mapping[idx]

        # 如果答案在上下文之外，设置答案位置为0
        if offset[context_start][0] > answer_end or offset[context_end][1] < answer_start:
            start_answer_token_pos = 0
            end_answer_token_pos = 0
        else:
            # 寻找答案开始位置的token索引
            _start = context_start
            while offset[_start][0] < answer_start and _start < context_end:
                _start += 1
            start_answer_token_pos = _start

            # 寻找答案结束位置的token索引
            _end = context_end
            while offset[_end][1] > answer_end and _end > context_start:
                _end -= 1
            end_answer_token_pos = _end

        # 将答案的token位置添加到列表中
        answer_starts.append(start_answer_token_pos)
        answer_ends.append(end_answer_token_pos)

        # 将数据ID添加到列表中
        data_ids.append(data['id'][data_overflow_to_sample_mapping[idx]])

        # 更新偏移映射，只保留上下文部分的偏移信息
        data_tokenized['offset_mapping'][idx] = [
            (o if data_tokenized.sequence_ids(idx)[k] == 1 else None)
            for k, o in enumerate(data_tokenized['offset_mapping'][idx])
        ]

    # 将数据ID、答案开始和结束位置添加到分词后的数据中
    data_tokenized['data_ids'] = data_ids
    data_tokenized["start_positions"] = answer_starts
    data_tokenized["end_positions"] = answer_ends

    # 返回处理后的数据
    return data_tokenized


def collate_fn(batch):
    input_ids = torch.tensor([example['input_ids'] for example in batch])
    attention_mask = torch.tensor([example['attention_mask'] for example in batch])
    token_type_ids = torch.tensor([example['token_type_ids'] for example in batch])
    start_positions = torch.tensor([example['start_positions'] for example in batch])
    end_positions = torch.tensor([example['end_positions'] for example in batch])
    data_overflow_to_sample_mapping = [example['overflow_to_sample_mapping'] for example in batch]
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "token_type_ids": token_type_ids.to(device),
        "start_positions": start_positions.to(device),
        "end_positions": end_positions.to(device),
        "overflow_to_sample_mapping": data_overflow_to_sample_mapping
    }


class BertForQADataset(Dataset):
    def __init__(self, data_path, split='train'):
        # 调用父类构造函数
        super().__init__()

        # 保存数据路径和分割（默认为'train'）
        self.data_path = data_path
        self.split = split

        # 加载数据集，并根据分割获取对应部分的数据
        self.data = load_dataset(data_path)[split]

        # 对数据进行预处理，这里假设data_process是一个预处理函数
        # batched=True表示可以对批量数据进行处理
        # remove_columns表示预处理后要移除的列名
        self.processed_data = self.data.map(data_process, batched=True, remove_columns=self.data.column_names)

    def __len__(self):
        # 返回数据集中的样本数量
        return len(self.processed_data)

    def __getitem__(self, idx):
        # 根据索引idx返回预处理后的数据样本
        return self.processed_data[idx]


if __name__ == '__main__':
    dataset = BertForQADataset('./cmrc2018', split='test')
    print(len(dataset))
    print(dataset[0])
    data_process(dataset.data[:2])

