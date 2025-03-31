import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from time import perf_counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .config import FedUnlConfig
from .sampler import GraphUnlSampler
from fedunlcrs.utils import FedUnlDataLoader, REC_METRIC_TABLE
from fedunlcrs.model import FedUnlMlp

class FedUnlWorker:
    def __init__(self, config:FedUnlConfig) -> None:
        self.config = config

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        mp.spawn(
            self.run_federate_unlearning,
            args=(),
            nprocs=self.config.n_proc,
            join=True,
        )

        return
    
    def run_federate_unlearning(self, rank:int) -> None:
        dist.init_process_group(
            backend="nccl",
            world_size=self.config.n_proc,
            rank=rank,
        )

        self.rank = rank
        self.device = f"cuda:{rank}"
        self.client_ids = list(range(
            rank * self.config.n_client_per_proc,
            (rank + 1) * self.config.n_client_per_proc,
        ))

        if self.rank == 0:
            self.sampler = GraphUnlSampler(self.config)
            os.makedirs(self.config.save_path, exist_ok=True)

        self.build_loader()
        self.build_model()

        for epoch in range(self.config.epochs):
            proc_train_time = []
            for (model, optimizer, dataloader) in zip(self.models, self.optims, self.dataloaders):
                client_train_time = self.federate(model, optimizer, dataloader)
                proc_train_time.append(client_train_time)
            proc_train_time = torch.tensor(proc_train_time).to(self.device)
            tot_train_time = [torch.zeros_like(proc_train_time).to(self.device) for _ in range(self.config.n_proc)]
            dist.gather(proc_train_time, tot_train_time if self.rank == 0 else None, 0)
            self.train_time = [tensor.item() for tensor in torch.concatenate(tot_train_time, dim=0).reshape(-1)]
            self.aggregate(mode="valid")
            self.unlearning()
        self.aggregate(mode="test")

        dist.destroy_process_group()

        return
    
    def build_loader(self) -> None:
        self.dataloaders : List[FedUnlDataLoader] = []
        for client_id in self.client_ids:
            self.dataloaders.append(
                FedUnlDataLoader(
                    self.config.dataset_name,
                    self.config.batch_size,
                    json.load(open(os.path.join(self.config.load_path, f"client_{client_id}_mask.json"), "r", encoding="utf-8")),
                    self.config.partition_mode,
                )
            )
        self.n_item = self.dataloaders[0].n_item
        self.n_entity = self.dataloaders[0].n_entity
        self.n_word = self.dataloaders[0].n_word
        return
    
    def build_model(self) -> None:
        self.models = []
        self.optims = []

        for _ in self.client_ids:
            if self.config.model_name == "mlp":
                client_model = FedUnlMlp(
                    self.n_item, self.n_entity, self.n_word,
                    self.config.mlp_config, self.device,
                ).to(self.device)
                client_optimizer = torch.optim.Adam(
                    client_model.parameters(),
                    lr=self.config.learning_rate,
                )
                self.models.append(client_model)
                self.optims.append(client_optimizer)

        return
    
    def aggregate(self, mode:str) -> None:
        self.evaluate(None, None, "reset")
        dataloader = self.dataloaders[0]
        for batch_data in tqdm(dataloader.get_data(mode, batch_size=self.config.batch_size), disable=(self.rank != 0)):
            proc_logits = []
            for client_model in self.models:
                logits, labels, loss = client_model.rec_forward(batch_data, dataloader.item_edger, dataloader.entity_edger, dataloader.word_edger)
                proc_logits.append(logits)
            
            proc_logits = torch.concatenate(proc_logits, dim=0)
            tot_logits  = [torch.zeros_like(proc_logits).to(self.device) for _ in range(self.config.n_proc)]
            dist.gather(proc_logits, tot_logits if self.rank == 0 else None, 0)
            tot_logits = torch.concatenate(tot_logits, dim=0)

            if self.rank == 0 and self.config.aggregate_methon == "mean":
                global_logits = tot_logits.mean(0, keepdim=True)
                ranks = torch.topk(global_logits, k=50, dim=-1)[1].detach().tolist()
                labels = [meta_data["label"] for meta_data in batch_data]
                self.evaluate(ranks, labels, "step")
        self.evaluate(None, None, "report")

        return
    
    def unlearning(self) -> None:
        if self.rank != 0:
            return
        
        unlearning_result = {}
        for (layer, topk) in self.config.unlearning_layer:
            [unlearning_clients, _] = self.sampler.sample(layer, topk, self.config.unlearning_sample_methon)
            unlearning_time = np.array([self.train_time[client_id] for client_id in unlearning_clients])
            unlearning_time_avg = float(unlearning_time.mean())
            unlearning_time_std = float(unlearning_time.std())
            unlearning_result[layer] = {
                "Avg": unlearning_time_avg,
                "Std": unlearning_time_std,
            }
        
        layers = [layer for (layer, topk) in self.config.unlearning_layer]
        unlearning_df = {"/": ["Avg", "Std"]}
        for layer in layers:
            unlearning_df[layer] = [
                f"{unlearning_result[layer]['Avg']:.4f}s",
                f"{unlearning_result[layer]['Std']:.4f}s",
            ]
        unlearning_df = pd.DataFrame(unlearning_df)
        print(unlearning_df)

        return
    
    def federate(self, model:torch.nn.Module, optimizer:torch.optim.Optimizer, dataloader:FedUnlDataLoader) -> float:
        start_time = perf_counter()
        for batch_data in tqdm(dataloader.get_data(mode="train", batch_size=self.config.batch_size), disable=(self.rank != 0)):
            optimizer.zero_grad()
            logits, labels, loss = model.rec_forward(batch_data, dataloader.item_edger, dataloader.entity_edger, dataloader.word_edger)
            loss.backward()
            optimizer.step()
        federate_time = perf_counter() - start_time
        return federate_time
    
    def evaluate(self, ranks:List[List[int]], labels:List[int], mode:str) -> None:
        if self.rank != 0:
            return

        if mode == "reset":
            for metrics in REC_METRIC_TABLE:
                for index in metrics:
                    metric = metrics[index]
                    metric.reset()
        elif mode == "step":
            for metrics in REC_METRIC_TABLE:
                for index in metrics:
                    metric = metrics[index]
                    for (rank, label) in zip(ranks, labels):
                        metric.step(rank, label)
        elif mode == "report":
            evaluate_res = []
            for metrics in REC_METRIC_TABLE:
                temp_res = {}
                for index in metrics:
                    metric = metrics[index]
                    temp_res[index] = f"{metric.report():.4f}"
                evaluate_res.append(temp_res)

            for meta_res in evaluate_res:
                meta_df = pd.DataFrame([meta_res])
                print(f"\n{meta_df.to_string(index=False)}")
        return