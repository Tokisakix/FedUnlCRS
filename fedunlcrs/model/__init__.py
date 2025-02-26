import torch
from typing import Dict

from .mlp import PretrainClassiferMLP

class PretrainEmbeddingModel(torch.nn.Module):
    def __init__(
            self, n_item:int, n_entity:int, n_word:int,
            embedding_dim:int, classifer:torch.nn.Module, device:str
        ) -> None:
        super().__init__()
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_word = n_word
        self.embedding_dim = embedding_dim
        self.item_embedding = torch.nn.Embedding(n_item, embedding_dim)
        self.entity_embedding = torch.nn.Embedding(n_entity, embedding_dim)
        self.word_embedding = torch.nn.Embedding(n_word, embedding_dim)
        self.classifer = classifer
        self.device = device
        return
    
    def forward(
            self, item_list:torch.LongTensor, entity_list:torch.LongTensor, word_list:torch.LongTensor,
            item_edger:Dict, entity_edger:Dict, word_edger:Dict
        ) -> torch.FloatTensor:
        item_emb = self.item_embedding(item_list).mean(0, keepdim=True) if item_list.shape[0] > 0 else torch.zeros((1, self.embedding_dim)).to(self.device)
        entity_emb = self.entity_embedding(entity_list).mean(0, keepdim=True) if entity_list.shape[0] > 0 else torch.zeros((1, self.embedding_dim)).to(self.device)
        word_emb = self.word_embedding(word_list).mean(0, keepdim=True) if word_list.shape[0] > 0 else torch.zeros((1, self.embedding_dim)).to(self.device)
        emb = (item_emb + entity_emb + word_emb) / 3.0
        out = self.classifer(emb, item_edger, entity_edger, word_edger)
        return out

def get_classifer(classifer_model:str) -> torch.nn.Module:
    classifer_model_table = {
        "mlp": PretrainClassiferMLP,
    }
    classifer = classifer_model_table[classifer_model]
    return classifer