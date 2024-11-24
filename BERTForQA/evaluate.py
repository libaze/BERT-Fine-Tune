# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/11/9 20:35
@File   : evaluate.py
@desc   : 
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.cmrc2018_evaluate import evaluate
from data.dataset import BertForQADataset, get_start_and_end_positions, collate_fn
from model import BertForQAModel
import time


def evaluate_model(model, dataset, dataloader):
    model.eval()
    start_logits = []
    end_logits = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            out = model(**batch)
            start_logits.append(out.start_logits)
            end_logits.append(out.end_logits)
        start_logits = torch.cat(start_logits, dim=0)
        end_logits = torch.cat(end_logits, dim=0)
        start_logits = start_logits.cpu().detach().numpy()
        end_logits = end_logits.cpu().detach().numpy()
    start_time = time.time()
    ground_true, predictions = get_start_and_end_positions(start_logits, end_logits, dataset, dataset.data)
    F1, EM, TOTAL, SKIP = evaluate(ground_true, predictions)
    print('Time: ', time.time() - start_time)
    AVG = (EM + F1) * 0.5
    print('EM score:', EM)
    print('F1 score:', F1)
    print('TOTAL score:', TOTAL)
    print('SKIP score:', SKIP)
    print('AVG score:', AVG)
    return AVG, F1, EM


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_qa = BertForQAModel(num_labels=2)
    bert_qa.load_state_dict(torch.load("./bert_qa_model1.bin"))
    bert_qa.to(device)
    dataset = BertForQADataset('./data/cmrc2018', split='test')
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    evaluate_model(bert_qa, dataset, dataloader)
















