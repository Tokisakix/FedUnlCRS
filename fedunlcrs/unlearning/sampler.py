import os
import json
import random
from typing import Dict, List, Tuple

from .config import FedUnlConfig

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
        
        return {
            "user": user_id_to_client,
            "conv": conv_id_to_client,
            "item": item_id_to_client,
            "entity": entity_id_to_client,
            "word": word_id_to_client,
        }
    
    def sample(self, rate:float, layer:str, methon:str) -> Tuple[List[int], List[int]]:
        id_to_client = self.id_to_community[layer]

        if methon == "random":
            unlearning_clients = self.methon_random(rate, id_to_client)

        all_clients = set(range(self.config.n_client))
        learning_clients = list(all_clients - set(unlearning_clients))

        return [learning_clients, unlearning_clients]
    
    def methon_random(self, rate:float, id_to_client:Dict) -> List[int]:
        all_ids = list(id_to_client.keys())
        sample_size = int(len(all_ids) * rate)
        selected_ids = random.sample(all_ids, sample_size)
        unlearning_clients = set()
        for id in selected_ids:
            unlearning_clients.add(id_to_client[id])
        return list(unlearning_clients)

class HyperGraphUnlSampler:
    def __init__(self, dataset_name:str, idx_to_client:Dict) -> None:
        raise NotImplementedError
    
    def sample(self, rate:float, layer:str, methon:str) -> Tuple[List[int], List[int]]:
        raise NotImplementedError