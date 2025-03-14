from tqdm import tqdm
from typing import Dict, List

import torch

from .base import BaseMetric
from .cov_metric import (
    ReCallMetric, MrrMetric, NdcgMetric,
    AprMetric, LtrMetric, CovMetric,
    GiniMetric, KlMetric, DiffMetric,
)

REC_METRIC_TABLE : List[Dict[str, BaseMetric]] = [
    {
        " REC@1 " : ReCallMetric(k=1),
        "REC@10 " : ReCallMetric(k=10),
        "REC@50 " : ReCallMetric(k=50),
        " MRR@1 " : MrrMetric(k=1),
        "MRR@10 " : MrrMetric(k=10),
        "MRR@50 " : MrrMetric(k=15),
        "NDCG@1 " : NdcgMetric(k=1),
        "NDCG@10" : NdcgMetric(k=10),
        "NDCG@50" : NdcgMetric(k=50),
    },
    {
        " APR@5 " : AprMetric(k=5),
        "APR@10 " : AprMetric(k=10),
        "APR@15 " : AprMetric(k=15),
        "APR@20 " : AprMetric(k=20),
        " LTR@5 " : LtrMetric(k=5),
        "LTR@10 " : LtrMetric(k=10),
        "LTR@15 " : LtrMetric(k=15),
        "LTR@20 " : LtrMetric(k=20),
        " Cov@5 " : CovMetric(k=5,  total_num=1),
        "Cov@10 " : CovMetric(k=10, total_num=1),
        "Cov@15 " : CovMetric(k=15, total_num=1),
        "Cov@20 " : CovMetric(k=20, total_num=1),
    },
    {
        "Gini@5 " : GiniMetric(k=5),
        "Gini@10" : GiniMetric(k=10),
        "Gini@15" : GiniMetric(k=15),
        "Gini@20" : GiniMetric(k=20),
        "  KL@5 " : KlMetric(k=5),
        " KL@10 " : KlMetric(k=10),
        " KL@15 " : KlMetric(k=15),
        " KL@20 " : KlMetric(k=20),
        "Diff@5 " : DiffMetric(k=5,  threshold=0.1),
        "Diff@10" : DiffMetric(k=10, threshold=0.1),
        "Diff@15" : DiffMetric(k=15, threshold=0.1),
        "Diff@20" : DiffMetric(k=20, threshold=0.1),
    },
]

def evaluate_rec(
        model:torch.nn.Module, dataloader:List,
        item_edger:Dict, entity_edger:Dict, word_edger:Dict
    ) -> List[Dict[str, float]]:
    device = model.device
    for metrics in REC_METRIC_TABLE:
        for index in metrics:
            metric = metrics[index]
            metric.reset()

    for meta_data in tqdm(dataloader):
        item_list = torch.LongTensor(meta_data["item"]).to(device)
        entity_list = torch.LongTensor(meta_data["entity"]).to(device)
        word_list = torch.LongTensor(meta_data["word"]).to(device)
        label = meta_data["label"][0]

        output = model(item_list, entity_list, word_list, item_edger, entity_edger, word_edger).detach()
        rank   = torch.topk(output, k=50, dim=-1)[1].tolist()[0]
        
        for metrics in REC_METRIC_TABLE:
            for index in metrics:
                metric = metrics[index]
                metric.step(rank, label)

    evaluate_res = []
    for metrics in REC_METRIC_TABLE:
        temp_res = {}
        for index in metrics:
            metric = metrics[index]
            temp_res[index] = f"{metric.report():.4f}"
        evaluate_res.append(temp_res)

    return evaluate_res