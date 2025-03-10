from typing import List

from .base import BaseMetric

class ReCallMetric(BaseMetric):
    def __init__(self, k:int) -> None:
        super().__init__()
        self.alpha :int = 0
        self.beta  :int = 0
        self.k     :int = k
        return
    
    def reset(self) -> None:
        self.alpha = 0
        self.beta  = 0
        return
    
    def step(self, rank: List[int], label: int) -> None:
        self.alpha += (1 if label in rank[:self.k] else 0)
        self.beta += 1
        return
    
    def report(self) -> float:
        res = self.alpha / self.beta
        return res