import os
import json
from typing import List

import torch.multiprocessing as mp

from .config import FedUnlConfig
from .sampler import GraphUnlSampler
from fedunlcrs.utils import FedUnlDataLoader
from fedunlcrs.model import FedUnlMlp

class FedUnlWorker:
    def __init__(self, config:FedUnlConfig) -> None:
        self.config = config

        mp.spawn(
            self.run_federate_unlearning,
            args=(),
            nprocs=self.config.n_proc,
            join=True,
        )

        return
    
    def run_federate_unlearning(self, rank:int) -> None:
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

        while True:
            pass

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

        for client_id in self.client_ids:
            if self.config.model_name == "mlp":
                self.models.append(
                    FedUnlMlp(
                        self.n_item, self.n_entity, self.n_word,
                        self.config.mlp_config, self.device,
                    ).to(self.device)
                )

        return
    
    def aggregate(self) -> None:
        raise NotImplementedError
    
    def unlearning(self, mode:str) -> None:
        raise NotImplementedError
    
    def federate(self, mode:str) -> None:
        raise NotImplementedError
    
    def close() -> None:
        return