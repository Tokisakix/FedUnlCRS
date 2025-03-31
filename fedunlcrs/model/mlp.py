import torch
from typing import Dict, List

class FedUnlMlp(torch.nn.Module):
    def __init__(self, n_item:int, n_entity:int, n_word:int, model_config:Dict, device:str):
        super().__init__()
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_word = n_word
        self.device = device
        self.hidden_dim = model_config["hidden_dim"]
        self.item_embedding = torch.nn.Embedding(self.n_item, self.hidden_dim)
        self.entity_embedding = torch.nn.Embedding(self.n_entity, self.hidden_dim)
        self.word_embedding = torch.nn.Embedding(self.n_word, self.hidden_dim)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            torch.nn.BatchNorm1d(self.hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            torch.nn.BatchNorm1d(self.hidden_dim), torch.nn.Softmax(dim=-1),
        )
        return
    
    def rec_forward(self, batch_data:List[Dict]) -> torch.FloatTensor:
        
        return