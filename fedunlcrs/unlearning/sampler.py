import os
import json
import random
from typing import Dict, List, Tuple

from .config import FedUnlConfig
from fedunlcrs.utils import get_dataset

class GraphUnlSampler:
    def __init__(self, config:FedUnlConfig) -> None:
        self.config = config
        self.item_popularity = {}
        self.entity_popularity = {}
        self.word_popularity = {}
        self.id_to_community = self.build_id_to_client()
        self.cal_hypergraph_popularity()
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

        # 超图内容
        self.item_hypergraph = []
        self.entity_hypergraph = []
        self.word_hypergraph = []
        
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
        self.item_hypergraph_id = 0
        self.entity_hypergraph_id = 0
        self.word_hypergraph_id = 0
        raw_train_dataset, _, _ = get_dataset(self.config.dataset_name)
        for conv in raw_train_dataset:
            conv_id = int(conv["conv_id"])
            if conv_id not in conv_id_to_client:
                continue
            client_id = conv_id_to_client[conv_id]

            # 记录具体的超图内容
            conv_item_list = set()
            conv_entity_list = set()
            conv_word_list = set()
            for dialog in conv["dialogs"]:
                meta_item_hypergraph = set()
                meta_entity_hypergraph = set()
                meta_word_hypergraph = set()
                for item in dialog["item"]:
                    if item not in conv_item_list:
                        item_hypergraph_id_to_client[self.item_hypergraph_id] = client_id
                        conv_item_list.add(item)
                        meta_item_hypergraph.add(item)
                        self.item_popularity[item] = self.item_popularity.get(item, 0) + 1
                        self.item_hypergraph_id += 1
                        self.item_hypergraph.append(list(meta_item_hypergraph))
                for entity in dialog["entity"]:
                    if entity not in conv_entity_list:
                        entity_hypergraph_id_to_client[self.entity_hypergraph_id] = client_id
                        conv_entity_list.add(entity)
                        meta_entity_hypergraph.add(entity)
                        self.entity_popularity[entity] = self.entity_popularity.get(entity, 0) + 1
                        self.entity_hypergraph_id += 1
                        self.entity_hypergraph.append(list(meta_entity_hypergraph))
                for word in dialog["word"]:
                    if word not in conv_word_list:
                        word_hypergraph_id_to_client[self.word_hypergraph_id] = client_id
                        conv_word_list.add(word)
                        meta_word_hypergraph.add(word)
                        self.word_popularity[word] = self.word_popularity.get(word, 0) + 1
                        self.word_hypergraph_id += 1
                        self.word_hypergraph.append(list(meta_word_hypergraph))

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
    
    def cal_hypergraph_popularity(self) -> None:
        assert len(self.item_hypergraph) == self.item_hypergraph_id
        assert len(self.entity_hypergraph) == self.entity_hypergraph_id
        assert len(self.word_hypergraph) == self.word_hypergraph_id

        def func(hypergraph:List[List[int]], popularity:Dict[int, int]) -> List[float]:
            hypergraph_popularity = []
            for meta_hypergraph in hypergraph:
                res = 0.0
                for item in meta_hypergraph:
                    res += popularity.get(item, 0)
                res /= len(meta_hypergraph)
                hypergraph_popularity.append(res)
            return hypergraph_popularity
        
        self.item_hypergraph_popularity = func(self.item_hypergraph, self.item_popularity)
        self.entity_hypergraph_popularity = func(self.entity_hypergraph, self.entity_popularity)
        self.word_hypergraph_popularity = func(self.word_hypergraph, self.word_popularity)
        return

    def sample(self, layer:str, topk:int, methon:str) -> Tuple[List[int], List[int], Dict]:
        id_to_client = self.id_to_community[layer]
        unlearning_mask = None

        if methon == "random":
            unlearning_clients, unlearning_mask = self.methon_random(topk, id_to_client)
        if methon == "topk":
            unlearning_clients, unlearning_mask = self.methon_topk(topk, id_to_client)

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
        sample_size = topk
        top_items = sorted(self.item_popularity.items(), key=lambda x: x[1], reverse=True)[:sample_size]
        top_item_ids = [item for item, _ in top_items]
        top_entity = sorted(self.entity_popularity.items(), key=lambda x: x[1], reverse=True)[:sample_size]
        top_entity_ids = [item for item, _ in top_entity]
        top_word = sorted(self.word_popularity.items(), key=lambda x: x[1], reverse=True)[:sample_size]
        top_word_ids = [item for item, _ in top_word]

        topk_ids = {
            "item": top_item_ids,
            "entity": top_entity_ids,
            "word": top_word_ids
        }
        selected_ids = {}
        all_ids = list(id_to_client.keys())
        sample_size = topk
        selected_ids = random.sample(all_ids, sample_size)
        unlearning_clients = set()
        for id in selected_ids:
            unlearning_clients.add(id_to_client[id])
        return [list(unlearning_clients), topk_ids]