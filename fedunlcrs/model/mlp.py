import torch
from typing import Dict

class PretrainClassiferMLP(torch.nn.Module):
    def __init__(self, embedding_dim:int, n_item:int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_item = n_item
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, n_item),
            torch.nn.Softmax(dim=-1)
        )
        return
    
    def forward(self, x:torch.FloatTensor, item_edger:Dict, entity_edger:Dict, word_edger:Dict) -> torch.FloatTensor:
        out = self.mlp(x)
        return out