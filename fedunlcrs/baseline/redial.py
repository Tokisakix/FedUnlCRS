import torch.nn as nn
from typing import Dict, List
import torch
import torch.nn.functional as F

class ReDialRecModel(torch.nn.Module):
    """

    Attributes:
        n_entity: A integer indicating the number of entities.
        layer_sizes: A integer indicating the size of layer in autorec.
        pad_entity_idx: A integer indicating the id of entity padding.

    """

    def __init__(self, n_item:int, n_entity:int, n_word:int, model_config:Dict, device:str):
        super(ReDialRecModel, self).__init__()
        self.device = device
        self.n_entity = n_entity
        self.layer_sizes = model_config['autorec_layer_sizes']
        self.pad_entity_idx = 0
        self.autorec_f = model_config['autorec_f']
        self.autorec_g = model_config['autorec_g']
        self.emb_dim = model_config["emb_dim"]
        self.n_item = n_item
        self.build_model()


    def build_model(self):
        # AutoRec
        if self.autorec_f == 'identity':
            self.f = lambda x: x
        elif self.autorec_f == 'sigmoid':
            self.f = nn.Sigmoid()
        elif self.autorec_f == 'relu':
            self.f = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for f : {}".format(self.opt['autorec_f']))

        if self.autorec_g == 'identity':
            self.g = lambda x: x
        elif self.autorec_g == 'sigmoid':
            self.g = nn.Sigmoid()
        elif self.autorec_g == 'relu':
            self.g = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for g : {}".format(self.opt['autorec_g']))

        self.encoder = nn.ModuleList([nn.Linear(self.n_entity, self.layer_sizes[0]) if i == 0
                                      else nn.Linear(self.layer_sizes[i - 1], self.layer_sizes[i])
                                      for i in range(len(self.layer_sizes))])
        self.user_repr_dim = self.layer_sizes[-1]
        self.decoder = nn.Linear(self.user_repr_dim, self.n_entity)
        self.loss = nn.CrossEntropyLoss()
        self.rec_bias = torch.nn.Linear(self.emb_dim, self.n_entity)
        self.entity_embedding = torch.nn.Embedding(self.n_item, self.emb_dim, 0)
        self.item_embedding = torch.nn.Embedding(self.n_item, self.emb_dim, 0)

    def rec_forward(self, batch_data:List[Dict], item_edger:Dict, entity_edger:Dict, word_edger:Dict):
        """

        Args:
            batch: ::

                {
                    'context_entities': (batch_size, n_entity),
                    'item': (batch_size)
                }

            mode (str)

        """
        related_entity = [meta_data["entity"] for meta_data in batch_data]
        related_item = [meta_data["item"] for meta_data in batch_data]
        labels = torch.LongTensor([meta_data["label"] for meta_data in batch_data]).to(self.device)

        entity_embedding = []
        item_embedding = []
        for entity_list, item_list in zip(related_entity, related_item):
            entity_tensor = torch.LongTensor(entity_list).to(self.device)
            entity_repr = self.entity_embedding(entity_tensor).mean(dim=0, keepdim=True)
            entity_embedding.append(entity_repr)

            item_tensor = torch.LongTensor(item_list).to(self.device)
            item_repr = self.item_embedding(item_tensor).mean(dim=0, keepdim=True)
            item_embedding.append(item_repr)

        entity_embedding = torch.concatenate(entity_embedding, dim=0)
        item_embedding = torch.concatenate(item_embedding, dim=0)
        combined_embedding = (entity_embedding + item_embedding) / 2
        logits = self.decoder(combined_embedding)
        loss = self.loss(logits, labels)
        return logits, labels, loss