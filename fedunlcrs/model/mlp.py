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
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim * 4, self.n_item),
            torch.nn.Softmax(dim=-1),
        )
        return
    
    def rec_forward(self, batch_data:List[Dict], item_edger:Dict, entity_edger:Dict, word_edger:Dict) -> torch.FloatTensor:
        label = []
        user_embedding = []

        for meta_data in batch_data:
            item_list = torch.LongTensor(meta_data["item"]).to(self.device)
            entity_list = torch.LongTensor(meta_data["entity"]).to(self.device)
            word_list = torch.LongTensor(meta_data["word"]).to(self.device)
            label.append(meta_data["label"])

            item_emb = self.item_embedding(item_list).mean(0, keepdim=True) if item_list.shape[0] > 0 else torch.zeros((1, self.hidden_dim)).to(self.device)
            entity_emb = self.entity_embedding(entity_list).mean(0, keepdim=True) if entity_list.shape[0] > 0 else torch.zeros((1, self.hidden_dim)).to(self.device)
            word_emb = self.word_embedding(word_list).mean(0, keepdim=True) if word_list.shape[0] > 0 else torch.zeros((1, self.hidden_dim)).to(self.device)
            user_embedding.append((item_emb + entity_emb + word_emb) / 3.0)

        user_embedding = torch.concatenate(user_embedding, dim=0)
        logits = self.fc(user_embedding)
        labels = torch.LongTensor(label).to(self.device)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        return logits, labels, loss