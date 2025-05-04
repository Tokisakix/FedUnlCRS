import heapq
import os
import json
import random
import pickle as pkl
from typing import Dict, List, Tuple

from .config import FedUnlConfig

class GraphUnlSampler:
    def __init__(self, config:FedUnlConfig) -> None:
        self.config = config
        (self.id_to_community,
        self.item_popularity, self.item_hypergraph_popularity,
        self.entity_popularity, self.entity_hypergraph_popularity,
        self.word_popularity, self.word_hypergraph_popularity) = \
        pkl.load(open(os.path.join(self.config.load_path, "community_sampler_data.pkl"), "rb"))
        self.unlearn_topk()
        return
    
    def unlearn_topk(self) -> None:
        sample_size = self.config.topk
        self.top_item_ids = [item for item, _ in heapq.nlargest(sample_size, self.item_popularity.items(), key=lambda x: x[1])]
        self.top_entity_ids = [item for item, _ in heapq.nlargest(sample_size, self.entity_popularity.items(), key=lambda x: x[1])]
        self.top_word_ids = [item for item, _ in heapq.nlargest(sample_size, self.word_popularity.items(), key=lambda x: x[1])]

        self.top_item_hypergraph_ids = heapq.nlargest(sample_size, self.item_hypergraph_popularity)
        self.top_entity_hypergraph_ids = heapq.nlargest(sample_size, self.entity_hypergraph_popularity)
        self.top_word_hypergraph_ids = heapq.nlargest(sample_size, self.word_hypergraph_popularity)
        return 

    def sample(self, layer:str, topk:int, methon:str) -> Tuple[List[int], List[int], Dict]:
        id_to_client = self.id_to_community[layer]
        unlearning_mask = None

        if methon == "random":
            unlearning_clients, unlearning_mask = self.methon_random(self.config.topk, id_to_client)
        if methon == "topk":
            unlearning_clients, unlearning_mask = self.methon_topk(self.config.topk, id_to_client)

        all_clients = set(range(self.config.n_client))
        learning_clients = list(all_clients - set(unlearning_clients))

        return unlearning_clients, learning_clients, unlearning_mask

    def methon_random(self, topk:int, id_to_client:Dict) -> Tuple[List[int], Dict]:
        sample_size = topk
        selected_ids = {}
        all_ids = list(id_to_client.keys())
        sample_size = topk
        selected_ids = random.sample(all_ids, sample_size)
        unlearning_clients = set()
        for id in selected_ids:
            unlearning_clients.add(id_to_client[id])
        return [list(unlearning_clients), None]
    
    def methon_topk(self, topk:int, id_to_client:Dict) -> Tuple[List[int], Dict]:
        topk_ids = {
            "item": self.top_item_ids,
            "entity": self.top_entity_ids,
            "word": self.top_word_ids,
            "item_hypergraph": self.top_item_hypergraph_ids,
            "entity_hypergraph": self.top_entity_hypergraph_ids,
            "word_hypergraph": self.top_word_hypergraph_ids
        }
        selected_ids = {}
        all_ids = list(id_to_client.keys())
        sample_size = topk
        selected_ids = random.sample(all_ids, sample_size)
        unlearning_clients = set()
        for id in selected_ids:
            unlearning_clients.add(id_to_client[id])
        return [list(unlearning_clients), topk_ids]