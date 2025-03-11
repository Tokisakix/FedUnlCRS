import math
from typing import Dict, List

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
    
class MrrMetric(BaseMetric):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.mrr_sum: float = 0.0
        self.beta: int = 0
        self.k: int = k
        return
    
    def reset(self) -> None:
        self.mrr_sum = 0.0
        self.beta = 0
        return
    
    def step(self, rank: List[int], label: int) -> None:
        for i, item in enumerate(rank[:self.k]):
            if item == label:
                self.mrr_sum += 1 / (i + 1)
                break
        self.beta += 1
        return
    
    def report(self) -> float:
        res = self.mrr_sum / self.beta
        return res


class NdcgMetric(BaseMetric):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.ndcg_sum: float = 0.0
        self.beta: int = 0
        self.k: int = k
        return
    
    def reset(self) -> None:
        self.ndcg_sum = 0.0
        self.beta = 0
        return
    
    def step(self, rank: List[int], label: int) -> None:
        dcg = 0.0
        idcg = 1.0
        for i, item in enumerate(rank[:self.k]):
            if item == label:
                dcg = 1 / (math.log2(i + 2))
                break
        self.ndcg_sum += dcg / idcg
        self.beta += 1
        return
    
    def report(self) -> float:
        res = self.ndcg_sum / self.beta
        return res
    
class AprMetric(BaseMetric):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.rank_set  :set = set()
        self.total_num :int = 0
        self.n :int = 0
        self.k :int = k
        return
    
    def reset(self) -> None:
        self.rank_set  :set = set()
        self.total_num :int = 0
        self.n :int = 0
        return
    
    def step(self, rank: List[int], label: int) -> None:
        self.total_num += self.k
        self.n += 1
        self.rank_set = set(list(self.rank_set) + rank[:self.k])
        return
    
    def report(self) -> float:
        res = self.total_num / (len(self.rank_set) * self.n)
        return res
    
class LtrMetric(BaseMetric):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.rank_set  :set  = set()
        self.rank_num  :Dict = {}
        self.total_num :int  = 0
        self.n :int = 0
        self.k :int = k
        return
    
    def reset(self) -> None:
        self.rank_set  :set = set()
        self.rank_num  :Dict = {}
        self.total_num :int = 0
        self.n :int = 0
        return
    
    def step(self, rank: List[int], label: int) -> None:
        self.total_num += self.k
        self.n += 1
        self.rank_set = set(list(self.rank_set) + rank[:self.k])
        for meta_rank in rank[:self.k]:
            self.rank_num[meta_rank] = self.rank_num.get(meta_rank, 0) + 1
        return
    
    def report(self) -> float:
        apr = self.total_num / (len(self.rank_set) * self.n)
        lt_num = 0
        for rank in self.rank_num:
            if self.rank_num[rank] >= apr * self.n:
                lt_num += 1
        res = lt_num / len(self.rank_num)
        return res
    
class CovMetric(BaseMetric):
    def __init__(self, k: int, total_num:int) -> None:
        super().__init__()
        self.cov_set  :set = set()
        self.total_num :int = total_num
        self.k :int = k
        return
    
    def reset(self) -> None:
        self.cov_set  :set = set()
        return
    
    def step(self, rank: List[int], label: int) -> None:
        self.cov_set = set(list(self.cov_set) + rank[:self.k])
        return
    
    def report(self) -> float:
        res = len(self.cov_set) / self.total_num
        return res

class GiniMetric(BaseMetric):
    pass

class KlMetric(BaseMetric):
    pass

class DiffMetric(BaseMetric):
    pass