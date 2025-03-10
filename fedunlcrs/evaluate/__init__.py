from tqdm import tqdm
from typing import Dict, List

import torch
import torch.distributed as dist

from .base import BaseMetric
from .cov_metric import ReCallMetric

REC_METRIC_TABLE : Dict[str, BaseMetric] = {
    "ReCall@1" : ReCallMetric(k=1),
    "ReCall@10": ReCallMetric(k=10),
    "ReCall@50": ReCallMetric(k=50),
}

def evaluate_rec(
        model:torch.nn.Module, dataloader:List,
        item_edger:Dict, entity_edger:Dict, word_edger:Dict
    ) -> Dict:
    device = model.device
    for index in REC_METRIC_TABLE:
        metric = REC_METRIC_TABLE[index]
        metric.reset()

    for meta_data in tqdm(dataloader):
        item_list = torch.LongTensor(meta_data["item"]).to(device)
        entity_list = torch.LongTensor(meta_data["entity"]).to(device)
        word_list = torch.LongTensor(meta_data["word"]).to(device)
        label = meta_data["label"][0]

        output = model(item_list, entity_list, word_list, item_edger, entity_edger, word_edger).detach()
        rank   = torch.topk(output, k=50, dim=-1)[1].tolist()[0]
        
        for index in REC_METRIC_TABLE:
            metric = REC_METRIC_TABLE[index]
            metric.step(rank, label)

    evaluate_res = {}
    for index in REC_METRIC_TABLE:
        metric = REC_METRIC_TABLE[index]
        evaluate_res[index] = f"{metric.report():.4f}"

    return evaluate_res