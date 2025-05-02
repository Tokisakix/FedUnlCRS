import os
import json
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
from time import perf_counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .config import FedUnlConfig
from .sampler import GraphUnlSampler
from fedunlcrs.utils import FedUnlDataLoader, REC_METRIC_TABLE
from fedunlcrs.model import FedUnlMlp, HyCoRec
from fedunlcrs.baseline import KBRDModel, BERTModel, GRU4RECModel, KGSFModel, NTRDModel, ReDialRecModel, TGRecModel, SASRECModel, TextCNNModel, MHIMModel

class FedUnlWorker:
    def __init__(self, config:FedUnlConfig) -> None:
        self.config = config
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        if self.config.n_client > 1:
            mp.spawn(
                self.run_federate_unlearning,
                args=(),
                nprocs=self.config.n_proc,
                join=True,
            )
        else:
            self.run_federate_unlearning(0)

        return
    
    def run_federate_unlearning(self, rank:int) -> None:
        if self.config.n_client > 1:
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
        self.full_data = {
            "model": self.config.model_name,
            "dataset": getattr(self.dataloaders[0], "dataset_name", "unknown"),
            "ablation_layer": getattr(self.config, "ablation_layer", None),
            "unlearning_methon": getattr(self.config, "unlearning_methon", None),
            "unlearning_num": getattr(self.config, "unlearning_num", None),
            "hyper_parameters": {
                "n_clients": self.config.n_client,
                "embedding_dim": self.config.emb_dim,
                "aggregate_rate": self.config.aggregate_rate
            },
            "evaluate_rec": [],
            "evaluate_cov": [],
            "unlearning_time": []
        }


        for epoch in range(self.config.epochs):
            proc_train_time = []
            for (model, optimizer, dataloader) in zip(self.models, self.optims, self.dataloaders):
                client_train_time = self.federate(model, optimizer, dataloader)
                proc_train_time.append(client_train_time)

            if self.config.n_client > 1:
                proc_train_time = torch.tensor(proc_train_time).to(self.device)
                tot_train_time = [torch.zeros_like(proc_train_time).to(self.device) for _ in range(self.config.n_proc)]
                dist.gather(proc_train_time, tot_train_time if self.rank == 0 else None, 0)
                self.train_time = [tensor.item() for tensor in torch.concatenate(tot_train_time, dim=0).reshape(-1)]
            else:
                self.train_time = proc_train_time

            self.evaluate(mode="valid")
            if self.config.n_client > 1:
                self.aggregate()
            self.unlearning()

        self.evaluate(mode="test")
        if self.config.n_client > 1:
            self.aggregate()

        if self.config.n_client > 1:
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
        self.models : List[torch.nn.Module] = []
        self.optims : List[torch.optim.Optimizer] = []

        for _ in self.client_ids:
            if self.config.model_name == "mlp":
                client_model = FedUnlMlp(
                    self.n_item, self.n_entity, self.n_word,
                    self.config.mlp_config, self.device,
                ).to(self.device)
            elif self.config.model_name == "hycorec":
                client_model = HyCoRec(
                    self.n_item, self.n_entity, self.n_word,
                    self.config.hycorec_config, self.device,
                ).to(self.device)
            elif self.config.model_name == "kbrd":
                client_model = KBRDModel(
                    self.n_item, self.n_entity, self.n_word,
                    self.config.kbrd_config, self.device,
                ).to(self.device)
            elif self.config.model_name == "bert":
                client_model = BERTModel(
                    self.n_item, self.n_entity, self.n_word,
                    self.config.bert_config, self.device,
                ).to(self.device)
            elif self.config.model_name == "gru4rec":
                client_model = GRU4RECModel(
                    self.n_item, self.n_entity, self.n_word,
                    self.config.gru4rec_config, self.device,
                ).to(self.device)
            elif self.config.model_name == "kgsf":
                client_model = KGSFModel(
                    self.n_item, self.n_entity, self.n_word,
                    self.config.kgsf_config, self.device,
                ).to(self.device)
            elif self.config.model_name == "ntrd":
                client_model = NTRDModel(
                    self.n_item, self.n_entity, self.n_word,
                    self.config.ntrd_config, self.device,
                ).to(self.device)
            elif self.config.model_name == "redial":
                client_model = ReDialRecModel(
                    self.n_item, self.n_entity, self.n_word,
                    self.config.redial_config, self.device,
                ).to(self.device)
            elif self.config.model_name == "sasrec":
                client_model = SASRECModel(
                    self.n_item, self.n_entity, self.n_word,
                    self.config.sasrec_config, self.device,
                ).to(self.device)
            elif self.config.model_name == "textcnn":
                client_model = TextCNNModel(
                    self.n_item, self.n_entity, self.n_word,
                    self.config.textcnn_config, self.device,
                ).to(self.device)
            elif self.config.model_name == "tgredial":
                client_model = TGRecModel(
                    self.n_item, self.n_entity, self.n_word,
                    self.config.tgredial_config, self.device,
                ).to(self.device)
            elif self.config.model_name == "mhim":
                client_model = MHIMModel(
                    self.n_item, self.n_entity, self.n_word,
                    self.config.mhim_config, self.device,
                ).to(self.device)
            client_optimizer = torch.optim.Adam(
                client_model.parameters(),
                lr=self.config.learning_rate,
            )
            self.models.append(client_model)
            self.optims.append(client_optimizer)

        return
    
    def aggregate(self) -> None:
        if self.config.aggregate_methon == "mean":
            proc_state_dict = copy.deepcopy(self.models[0].state_dict())
            for model_id in range(1, len(self.models)):
                client_state_dict = self.models[model_id].state_dict()
                for param_key in proc_state_dict.keys():
                    proc_state_dict[param_key] += client_state_dict[param_key]
            
            for param_key in proc_state_dict.keys():
                proc_state_dict[param_key] /= self.config.n_client_per_proc
                dist.all_reduce(proc_state_dict[param_key])
                proc_state_dict[param_key] /= self.config.n_proc

            for model in self.models:
                new_client_state_dict = copy.deepcopy(model.state_dict())
                for param_key in new_client_state_dict.keys():
                    new_client_state_dict[param_key] += self.config.aggregate_rate * (proc_state_dict[param_key] - new_client_state_dict[param_key])
                model.load_state_dict(new_client_state_dict)

        for (client_id, model) in zip(self.client_ids, self.models):
            torch.save(
                model.state_dict(),
                open(os.path.join(self.config.save_path, f"client_{client_id}_state_dict.pth"), "wb"),
            )

        if self.rank == 0:
            torch.save(
                proc_state_dict,
                open(os.path.join(self.config.save_path, f"global_{self.config.aggregate_methon}_state_dict.pth"), "wb"),
            )

        return
    
    def unlearning(self) -> Dict:
        if self.rank != 0:
            return
        
        time_entry = {
            "epoch": self.config.epochs,
            "user": {"time_avg": 0.0, "time_std": 0.0},
            "conv": {"time_avg": 0.0, "time_std": 0.0},
            "item": {"time_avg": 0.0, "time_std": 0.0},
            "entity": {"time_avg": 0.0, "time_std": 0.0},
            "word": {"time_avg": 0.0, "time_std": 0.0},
            "item_hypergraph": {"time_avg": 0.0, "time_std": 0.0},
            "entity_hypergraph": {"time_avg": 0.0, "time_std": 0.0},
            "word_hypergraph": {"time_avg": 0.0, "time_std": 0.0}
        }

        unlearning_result = {}
        for (layer, topk) in self.config.unlearning_layer:
            unlearning_clients, _, unlearning_mask = self.sampler.sample(
                layer, topk, self.config.unlearning_sample_methon)
            unlearning_time = np.array([self.train_time[client_id] for client_id in unlearning_clients])
            unlearning_result[layer] = {
                "Avg": float(unlearning_time.mean()),
                "Std": float(unlearning_time.std())
            }

        layers = [layer for (layer, topk) in self.config.unlearning_layer]
        unlearning_df = {"/": ["Avg", "Std"]}
        for layer in layers:
            unlearning_df[layer] = [
                f"{unlearning_result[layer]['Avg']:.4f}s",
                f"{unlearning_result[layer]['Std']:.4f}s",
            ]
            time_entry[layer]["time_avg"] = unlearning_result[layer]["Avg"]
            time_entry[layer]["time_std"] = unlearning_result[layer]["Std"]
        
        print(pd.DataFrame(unlearning_df))
        os.makedirs(self.config.evaluate_path, exist_ok=True)
        save_path = os.path.join(self.config.evaluate_path, "test_evaluation_result.json")
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                with open(save_path, "r") as f:
                 full_data = json.load(f)
        else:
            full_data = {
                "model": self.config.model_name,
                "dataset": getattr(self.dataloaders[0], "dataset_name", "unknown"),
                "ablation_layer": getattr(self.config, "ablation_layer", None),
                "unlearning_methon": getattr(self.config, "unlearning_methon", None),
                "unlearning_num": getattr(self.config, "unlearning_num", None),
                "hyper_parameters": {
                    "n_clients": self.config.n_client,
                    "embedding_dim": self.config.emb_dim,
                    "aggregate_rate": self.config.aggregate_rate
                },
                "evaluate_rec": {},
                "evaluate_cov": {},
                "unlearning_time": {}
            }
        full_data["unlearning_time"]= time_entry

        with open(save_path, "w") as f:
            json.dump(full_data, f, indent=4)

        return unlearning_mask
    
    def federate(self, model:torch.nn.Module, optimizer:torch.optim.Optimizer, dataloader:FedUnlDataLoader) -> float:
        start_time = perf_counter()
        for batch_data in tqdm(dataloader.get_data(mode="train", batch_size=self.config.batch_size), disable=(self.rank != 0)):
            optimizer.zero_grad()
            logits, labels, loss = model.rec_forward(batch_data, dataloader.item_edger, dataloader.entity_edger, dataloader.word_edger)
            loss.backward()
            optimizer.step()
        federate_time = perf_counter() - start_time
        return federate_time
    
    def evaluate(self, mode: str, unlearning_mask: Dict = None, unlearning: bool = False) -> None:
        proc_evaluate_res = []

        for model, dataloader in zip(self.models, self.dataloaders):
            client_result = self.evaluate_rec(model, dataloader, mode, unlearning_mask)
            proc_evaluate_res.append(client_result)

        if self.config.n_client > 1:
            proc_evaluate_res = torch.tensor(proc_evaluate_res).to(self.device)
            gathered = [torch.zeros_like(proc_evaluate_res).to(self.device) for _ in range(self.config.n_proc)]
            dist.gather(proc_evaluate_res, gathered if self.rank == 0 else None, dst=0)

            if self.rank != 0:
                return

            evaluate_res = torch.cat(gathered, dim=0).reshape((self.config.n_client, -1)).cpu().numpy()
        else:
            evaluate_res = np.array(proc_evaluate_res)

        if self.rank == 0:
            evaluate_res_avg = evaluate_res.mean(axis=0)
            evaluate_res_std = evaluate_res.std(axis=0)

            rec_task = {}
            fairness_aware = {}
            metric_idx = 0

            for metric_group in REC_METRIC_TABLE:
                for metric_name in metric_group.keys():
                    value = round(float(evaluate_res_avg[metric_idx]), 4)
                    if any(prefix in metric_name for prefix in ['APR', 'LTR', 'Cov', 'Gini', 'KL', 'Diff']):
                        fairness_aware[metric_name] = value
                    else:
                        rec_task[metric_name] = value
                    metric_idx += 1

            result_entry = {
                "epoch": self.config.epochs,
                "mode": mode,
                "unlearning": unlearning,
                "rec_task": rec_task,
                "fairness_aware": fairness_aware
            }

            os.makedirs(self.config.evaluate_path, exist_ok=True)
            save_path = os.path.join(self.config.evaluate_path, f"{mode}_evaluation_result.json")

            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                with open(save_path, "r") as f:
                 full_data = json.load(f)
            else:
                full_data = {
                    "model": self.config.model_name,
                    "dataset": getattr(self.dataloaders[0], "dataset_name", "unknown"),
                    "ablation_layer": getattr(self.config, "ablation_layer", None),
                    "unlearning_methon": getattr(self.config, "unlearning_methon", None),
                    "unlearning_num": getattr(self.config, "unlearning_num", None),
                    "hyper_parameters": {
                        "n_clients": self.config.n_client,
                        "embedding_dim": self.config.emb_dim,
                        "aggregate_rate": self.config.aggregate_rate
                    },
                    "evaluate_rec": {},
                    "evaluate_cov": {},
                    "unlearning_time": {}
                }

            full_data["evaluate_rec"]= result_entry

            with open(save_path, "w") as f:
                json.dump(full_data, f, indent=4)

            print(f"Evaluation result saved to {save_path}")
        
    def evaluate_rec(self, model:torch.nn.Module, dataloader:FedUnlDataLoader, mode:str, unlearning_mask:Dict=None) -> List[float]:
        for metrics in REC_METRIC_TABLE:
            for index in metrics:
                metric = metrics[index]
                metric.reset()

        for batch_data in tqdm(dataloader.get_data(mode, self.config.batch_size, unlearning_mask), disable=(self.rank != 0)):
            labels = [meta_data["label"] for meta_data in batch_data]
            logits, _, loss = model.rec_forward(batch_data, dataloader.item_edger, dataloader.entity_edger, dataloader.word_edger)
            ranks  = torch.topk(logits, k=50, dim=-1)[1].tolist()
            
            for metrics in REC_METRIC_TABLE:
                for index in metrics:
                    metric = metrics[index]
                    for (rank, label) in zip(ranks, labels):
                        metric.step(rank, label)

        evaluate_res = []
        for metrics in REC_METRIC_TABLE:
            for index in metrics:
                metric = metrics[index]
                evaluate_res.append(metric.report())

        return evaluate_res