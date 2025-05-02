import torch.nn as nn
from typing import Dict, List
import torch


class ReDialRecModel(torch.nn.Module):
    """

    Attributes:
        n_entity: A integer indicating the number of entities.
        layer_sizes: A integer indicating the size of layer in autorec.
        pad_entity_idx: A integer indicating the id of entity padding.

    """

    def __init__(self, n_item:int, n_entity:int, n_word:int, model_config:Dict, device:str):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        super(ReDialRecModel, self).__init__()
        self.n_entity = n_entity
        self.layer_sizes = model_config['autorec_layer_sizes']
        self.pad_entity_idx = 0


    def build_model(self):
        # AutoRec
        if self.opt['autorec_f'] == 'identity':
            self.f = lambda x: x
        elif self.opt['autorec_f'] == 'sigmoid':
            self.f = nn.Sigmoid()
        elif self.opt['autorec_f'] == 'relu':
            self.f = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for f : {}".format(self.opt['autorec_f']))

        if self.opt['autorec_g'] == 'identity':
            self.g = lambda x: x
        elif self.opt['autorec_g'] == 'sigmoid':
            self.g = nn.Sigmoid()
        elif self.opt['autorec_g'] == 'relu':
            self.g = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for g : {}".format(self.opt['autorec_g']))

        self.encoder = nn.ModuleList([nn.Linear(self.n_entity, self.layer_sizes[0]) if i == 0
                                      else nn.Linear(self.layer_sizes[i - 1], self.layer_sizes[i])
                                      for i in range(len(self.layer_sizes))])
        self.user_repr_dim = self.layer_sizes[-1]
        self.decoder = nn.Linear(self.user_repr_dim, self.n_entity)
        self.loss = nn.CrossEntropyLoss()

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
        for i, layer in enumerate(self.encoder):
            related_entity = self.f(layer(related_entity))
        scores = self.g(self.decoder(related_entity))
        loss = self.loss(scores, related_item)

        return scores, loss, loss