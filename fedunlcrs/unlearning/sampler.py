import os
import json
import random
from typing import Dict, List, Tuple

from .config import FedUnlConfig
from fedunlcrs.utils import get_dataset

class GraphUnlSampler:
    def __init__(self, config:FedUnlConfig) -> None:
        self.config = config
        self.id_to_community = self.build_id_to_client()
        return
    
    def build_id_to_client(self) -> Dict:
        user_id_to_client = {}
        conv_id_to_client = {}
        item_id_to_client = {}
        entity_id_to_client = {}
        word_id_to_client = {}
        item_hypergraph_id_to_client = {}
        entity_hypergraph_id_to_client = {}
        word_hypergraph_id_to_client = {}
        
        # build graph unlearning id to client
        for i in range(self.config.n_client):
            file_path = os.path.join(self.config.load_path, f"client_{i}_mask.json")
            with open(file_path, "r", encoding="utf-8") as f:
                client_data = json.load(f)
            
            for user_id in client_data.get("user_mask", []):
                user_id_to_client[user_id] = i
            
            for conv_id in client_data.get("conv_mask", []):
                conv_id_to_client[conv_id] = i
            
            for item_id in client_data.get("item_mask", []):
                item_id_to_client[item_id] = i
            
            for entity_id in client_data.get("entity_mask", []):
                entity_id_to_client[entity_id] = i
            
            for word_id in client_data.get("word_mask", []):
                word_id_to_client[word_id] = i
        
        # build hypergraph unlearning id to client
        item_hypergraph_id = 0
        entity_hypergraph_id = 0
        word_hypergraph_id = 0
        raw_train_dataset, _, _ = get_dataset(self.config.dataset_name)
        for conv in raw_train_dataset:
            conv_id = int(conv["conv_id"])
            if conv_id not in conv_id_to_client:
                continue
            client_id = conv_id_to_client[conv_id]

            conv_item_list = set()
            conv_entity_list = set()
            conv_word_list = set()
            for dialog in conv["dialogs"]:
                for item in dialog["item"]:
                    if item not in conv_item_list:
                        item_hypergraph_id_to_client[item_hypergraph_id] = client_id
                        conv_item_list.add(item)
                        item_hypergraph_id += 1
                for entity in dialog["entity"]:
                    if entity not in conv_entity_list:
                        entity_hypergraph_id_to_client[entity_hypergraph_id] = client_id
                        conv_entity_list.add(entity)
                        entity_hypergraph_id += 1
                for word in dialog["word"]:
                    if word not in conv_word_list:
                        word_hypergraph_id_to_client[word_hypergraph_id] = client_id
                        conv_word_list.add(word)
                        word_hypergraph_id += 1

        return {
            "user": user_id_to_client,
            "conv": conv_id_to_client,
            "item": item_id_to_client,
            "entity": entity_id_to_client,
            "word": word_id_to_client,
            "item_hypergraph": item_hypergraph_id_to_client,
            "entity_hypergraph": entity_hypergraph_id_to_client,
            "word_hypergraph": word_hypergraph_id_to_client,
        }
    
    def sample(self, layer:str, topk:int, methon:str) -> Tuple[List[int], List[int]]:
        id_to_client = self.id_to_community[layer]

        if methon == "random":
            unlearning_clients = self.methon_random(topk, id_to_client)

        all_clients = set(range(self.config.n_client))
        learning_clients = list(all_clients - set(unlearning_clients))

        return [learning_clients, unlearning_clients]
    
    def methon_random(self, topk:int, id_to_client:Dict) -> List[int]:
        all_ids = list(id_to_client.keys())
        sample_size = topk
        selected_ids = random.sample(all_ids, sample_size)
        unlearning_clients = set()
        for id in selected_ids:
            unlearning_clients.add(id_to_client[id])
        return list(unlearning_clients)