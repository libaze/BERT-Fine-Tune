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


# def get_start_and_end_positions(start_logits, end_logits, processed_data, examples):
#     ids_to_processed_data = defaultdict(list)
#     for idx in range(len(processed_data)):
#         ids_to_processed_data[processed_data[idx]['data_ids']].append(idx)
#     n_best = 5
#     max_answer_length = 30
#     predictions = []
#     ground_true = []
#     for example in tqdm(examples):
#         _id = example['id']
#         context = example["context"]
#         answers = []
#         for idx in ids_to_processed_data[_id]:
#             start_logit = start_logits[idx]
#             end_logit = end_logits[idx]
#             offset = processed_data['offset_mapping'][idx]
#             start_indexes = np.argsort(start_logit)[::-1][:n_best].tolist()
#             end_indexes = np.argsort(end_logit)[::-1][:n_best].tolist()
#             for start_index in start_indexes:
#                 for end_index in end_indexes:
#                     if offset[start_index] is None or offset[end_index] is None:    # 答案不能落在问题里
#                         continue
#                     if end_index < start_index or end_index - start_index + 1 > max_answer_length:
#                         continue
#                     answers.append({
#                         "text": context[offset[start_index][0]: offset[end_index][1]],
#                         "score": start_logit[start_index] + end_logit[end_index]
#                     })
#         if len(answers) > 0:
#             best_answer = max(answers, key=lambda x: x["score"])
#             predictions.append(best_answer["text"])
#         else:
#             predictions.append('')
#         ground_true.append(example["answers"]["text"])
#
#     return ground_true, predictions


def process_example(example, ids_to_processed_data, data_offset_mapping, start_logits, end_logits, n_best=5, max_answer_length=30):
    _id = example['id']
    context = example["context"]
    answers = []
    for idx in ids_to_processed_data[_id]:
        start_logit = start_logits[idx]
        end_logit = end_logits[idx]
        offset = data_offset_mapping[idx]
        start_indexes = np.argsort(start_logit)[::-1][:n_best].tolist()
        end_indexes = np.argsort(end_logit)[::-1][:n_best].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                if offset[start_index] is None or offset[end_index] is None:
                    continue
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue
                answers.append({
                    "text": context[offset[start_index][0]: offset[end_index][1]],
                    "score": start_logit[start_index] + end_logit[end_index]
                })
    if len(answers) > 0:
        best_answer = max(answers, key=lambda x: x["score"])
        return example["answers"]["text"], best_answer["text"]
    else:
        return example["answers"]["text"], ''


def get_start_and_end_positions(start_logits, end_logits, processed_data, examples, max_workers=4):
    ids_to_processed_data = defaultdict(list)
    for idx, item in enumerate(processed_data):
        ids_to_processed_data[item['data_ids']].append(idx)
    data_offset_mapping = processed_data['offset_mapping']
    n_best = 5
    max_answer_length = 30
    ground_true = []
    predictions = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_example, example, ids_to_processed_data, data_offset_mapping, start_logits, end_logits, n_best, max_answer_length) for example in examples]
        for future in concurrent.futures.as_completed(futures):
            gt, pred = future.result()
            ground_true.append(gt)
            predictions.append(pred)

    return ground_true, predictions


def data_process(data):
    data_tokenized = tokenizer(
        text=data['question'],
        text_pair=data['context'],
        max_length=384,
        truncation='only_second',
        stride=128,
        padding='max_length',
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    data_overflow_to_sample_mapping = data_tokenized['overflow_to_sample_mapping']
    data_offset_mapping = data_tokenized['offset_mapping']
    data_token_type_ids = data_tokenized['token_type_ids']
    answer_starts = []
    answer_ends = []
    data_ids = []
    for idx, _ in enumerate(data_overflow_to_sample_mapping):
        answer_start = data['answers'][data_overflow_to_sample_mapping[idx]]['answer_start'][0]
        answer_end = answer_start + len(data['answers'][data_overflow_to_sample_mapping[idx]]['text'][0])
        context_start = data_token_type_ids[idx].index(1)   # 上下文开始的位置
        context_end = data_tokenized.sequence_ids(idx)[context_start:].index(None) - 1    # 上下文结束的位置
        offset = data_offset_mapping[idx]
        if offset[context_start][0] > answer_end or offset[context_end][0] < answer_start:
            start_answer_token_pos = 0
            end_answer_token_pos = 0
        else:
            _start = context_start
            while offset[_start][0] < answer_start and _start < context_end:
                _start += 1
            start_answer_token_pos = _start
            _end = context_end
            while offset[_end][1] > answer_end and _end > context_start:
                _end -= 1
            end_answer_token_pos = _end
        answer_starts.append(start_answer_token_pos)
        answer_ends.append(end_answer_token_pos)
        data_ids.append(data['id'][data_overflow_to_sample_mapping[idx]])
        data_tokenized['offset_mapping'][idx] = [
            (o if data_tokenized.sequence_ids(idx)[k] == 1 else None)
            for k, o in enumerate(data_tokenized['offset_mapping'][idx])
        ]

    data_tokenized['data_ids'] = data_ids
    data_tokenized["start_positions"] = answer_starts
    data_tokenized["end_positions"] = answer_ends
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
        super().__init__()
        self.data_path = data_path
        self.data = load_dataset(data_path)[split]
        # self.data = load_dataset(data_path)[split].select(range(100))
        self.processed_data = self.data.map(data_process, batched=True, remove_columns=self.data.column_names)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


if __name__ == '__main__':
    dataset = BertForQADataset('./cmrc2018', split='test')
    print(len(dataset))
    print(dataset[0])

