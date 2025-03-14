import math
import numpy as np
from numba import njit, prange
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

@njit(parallel=True)
def cal_Gini_njit(n, freq):
    res = 0
    for i in prange(n):
        for j in range(n):
            res += abs(freq[i] - freq[j])
    return res

class GiniMetric(BaseMetric):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.freq  :Dict = {}
        self.n :int = 0
        self.k :int = k
        return
    
    def reset(self) -> None:
        self.freq  :Dict = {}
        return
    
    def step(self, rank: List[int], label: int) -> None:
        self.n += self.k
        for item in rank[:self.k]:
            self.freq[item] = self.freq.get(item, 0) + 1
        return
    
    def report(self) -> float:
        for item in self.freq:
            self.freq[item] /= self.n

        avg_freq = sum(self.freq.values()) / len(self.freq)
        freq     = np.array(list(self.freq.values()), dtype=np.float64)
        res      = cal_Gini_njit(len(self.freq), freq)
        res      /= (2 * len(self.freq) ** 2 * avg_freq)
        return res

class KlMetric(BaseMetric):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.d1  :Dict = {}
        self.n :int = 0
        self.k :int = k
        return
    
    def reset(self) -> None:
        self.d1  :Dict = {}
        return
    
    def step(self, rank: List[int], label: int) -> None:
        self.n += self.k
        for item in rank[:self.k]:
            self.d1[item] = self.d1.get(item, 0) + 1
        return
    
    def report(self) -> float:
        for item in self.d1:
            self.d1[item] /= self.n
        d2 = {item: 1 / self.n for item in self.d1}

        res = 0
        for item in self.d1:
            if self.d1[item] > 0 and d2.get(item, 0) > 0:
                res += self.d1[item] * math.log(self.d1[item] / d2[item])
        return res

@njit(parallel=True)
def cal_Difference_njit(n, pred_scores, threshold):
    diff_count = 0
    for i in prange(n):
        for j in range(i + 1, n):
            if abs(pred_scores[i] - pred_scores[j]) < threshold:
                diff_count += 1
    return diff_count / (n * (n - 1) / 2)

class DiffMetric(BaseMetric):
    def __init__(self, k: int, threshold:float) -> None:
        super().__init__()
        self.all_item :List = []
        self.k :int = k
        self.threshold :float = threshold
        return
    
    def reset(self) -> None:
        self.all_item :List = []
        return
    
    def step(self, rank: List[int], label: int) -> None:
        self.all_item += rank[:self.k]
        return
    
    def report(self) -> float:
        pred_scores_dict = {item: 1 / (index + 1) for index, item in enumerate(set(self.all_item))}
        pred_scores      = np.array([pred_scores_dict[item] for item in self.all_item], dtype=np.float64)
        res = cal_Difference_njit(len(self.all_item), pred_scores, self.threshold)
        return res