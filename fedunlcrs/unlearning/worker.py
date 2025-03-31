import os

from .config import FedUnlConfig
from .sampler import GraphUnlSampler
from fedunlcrs.utils import FedUnlDataLoader

class FedUnlWorker:
    def __init__(self, config:FedUnlConfig) -> None:
        self.config = config
        self.sampler = GraphUnlSampler(self.config)

        os.makedirs(self.config.save_path, exist_ok=True)

        return
    
    def build_model(self) -> None:
        raise NotImplementedError
    
    def aggregate(self) -> None:
        raise NotImplementedError
    
    def unlearning(self, mode:str) -> None:
        raise NotImplementedError
    
    def federate(self, mode:str) -> None:
        raise NotImplementedError