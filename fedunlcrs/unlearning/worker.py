from .config import FedUnlConfig

class FedUnlWorker:
    def __init__(self, config:FedUnlConfig):
        raise NotImplementedError
    
    def build_model(self) -> None:
        raise NotImplementedError
    
    def aggregate(self) -> None:
        raise NotImplementedError
    
    def unlearning(self, mode:str) -> None:
        raise NotImplementedError
    
    def federate(self, mode:str) -> None:
        raise NotImplementedError