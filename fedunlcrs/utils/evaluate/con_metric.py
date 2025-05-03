from typing import List
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from .base import BaseMetric

class BleuMetric(BaseMetric):
    def __init__(self, k:int) -> None:
        super().__init__()
        self.k     :int = k
        self.weight = {
            1: (1.0, ),
            2: (0.5, 0.5),
            3: (1/3, 1/3, 1/3),
            4: (0.25, 0.25, 0.25, 0.25),
        }[k]
        self.smoothie = SmoothingFunction().method1
        self.alpha = 0.0
        self.beta = 0.0
        return
    
    def reset(self) -> None:
        self.alpha = 0.0
        self.beta = 0.0
        return
    
    def step(self, reponse: List[int], label: List[int]) -> None:
        bleu = sentence_bleu(
            [label], reponse,
            weights=self.weight,
            smoothing_function=self.smoothie
        )
        self.alpha += bleu
        self.beta += 1.0
        return
    
    def report(self) -> float:
        return self.alpha / self.beta

class DistMetric(BaseMetric):
    def __init__(self, k:int) -> None:
        super().__init__()
        self.k     :int = k
        self.alpha = 0.0
        self.beta = 0.0
        return
    
    def reset(self) -> None:
        self.alpha = 0.0
        self.beta = 0.0
        return
    
    def step(self, reponse: List[str], label: List[str]) -> None:
        if len(reponse) < self.k:
            dist = 0.0
        else:
            total_ngrams = list(ngrams(reponse, self.k))
            unique_ngrams = set(total_ngrams)
            dist = len(unique_ngrams) / len(total_ngrams)
        self.alpha += dist
        self.beta += 1.0
        return
    
    def report(self) -> float:
        return self.alpha / self.beta