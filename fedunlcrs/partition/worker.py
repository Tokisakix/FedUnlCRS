import os
import json
import random
import pickle as pkl
from typing import List, Dict

from .config import PartitionConfig
from fedunlcrs.utils import FedUnlDataLoader

class PartitionWorker:
    def __init__(self, config:PartitionConfig) -> None:
        os.makedirs(config.save_dir, exist_ok=True)
        self.config = config
        self.dataloader = FedUnlDataLoader(self.config.dataset_name, 1, 0, None, None, None)
        self.raw_train_dataset = self.dataloader.raw_train_dataset

        self.item_popularity = {}
        self.entity_popularity = {}
        self.word_popularity = {}

        if self.config.partition_methon == "random":
            self.methon_random()

        self.id_to_community = self.build_id_to_client()
        (self.item_hypergraph_popularity, self.entity_hypergraph_popularity, self.word_hypergraph_popularity) = self.cal_hypergraph_popularity()
        pkl.dump(
            (self.id_to_community,
            self.item_popularity, self.item_hypergraph_popularity,
            self.entity_popularity, self.entity_hypergraph_popularity,
            self.word_popularity, self.word_hypergraph_popularity),
            open(os.path.join(self.config.save_dir, f"community_sampler_data.pkl"), "wb")
        )

        return
    
    def methon_random(self) -> None:
        user_id_list = set()
        conv_id_list = set()
        item_id_list = list(range(self.dataloader.n_item))
        entity_id_list = list(range(self.dataloader.n_entity))
        word_id_list = list(range(self.dataloader.n_word))

        for dataset in [self.dataloader.train_dataset, self.dataloader.valid_dataset, self.dataloader.test_dataset]:
            for conv in dataset:
                user_id_list.add(conv["user_id"])
                conv_id_list.add(conv["conv_id"])
        user_id_list = list(user_id_list)
        conv_id_list = list(conv_id_list)

        def split_into_clients(data_list, n_clients):
            random.shuffle(data_list)
            return [data_list[i::n_clients] for i in range(n_clients)]

        user_splits = split_into_clients(user_id_list, self.config.partition_num)
        conv_splits = split_into_clients(conv_id_list, self.config.partition_num)
        item_splits = split_into_clients(item_id_list, self.config.partition_num)
        entity_splits = split_into_clients(entity_id_list, self.config.partition_num)
        word_splits = split_into_clients(word_id_list, self.config.partition_num)

        for i in range(self.config.partition_num):
            client_data = {
                "user_mask": user_splits[i],
                "conv_mask": conv_splits[i],
                "item_mask": item_splits[i],
                "entity_mask": entity_splits[i],
                "word_mask": word_splits[i],
            }
            file_path = os.path.join(self.config.save_dir, f"client_{i}_mask.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(client_data, f, ensure_ascii=False, indent=4)

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

        self.item_hypergraph = []
        self.entity_hypergraph = []
        self.word_hypergraph = []
        
        for i in range(self.config.partition_num):
            file_path = os.path.join(self.config.save_dir, f"client_{i}_mask.json")
            with open(file_path, "r", encoding="utf-8") as f:
                client_data = json.load(f)
            client_data_get = client_data.get
            for user_id in client_data_get("user_mask", []):
                user_id_to_client[user_id] = i
            
            for conv_id in client_data_get("conv_mask", []):
                conv_id_to_client[conv_id] = i
            
            for item_id in client_data_get("item_mask", []):
                item_id_to_client[item_id] = i
            
            for entity_id in client_data_get("entity_mask", []):
                entity_id_to_client[entity_id] = i
            
            for word_id in client_data_get("word_mask", []):
                word_id_to_client[word_id] = i
        
        self.item_hypergraph_id = 0
        self.entity_hypergraph_id = 0
        self.word_hypergraph_id = 0
        for conv in self.raw_train_dataset:
            conv_id = int(conv["conv_id"])
            if conv_id not in conv_id_to_client:
                continue
            client_id = conv_id_to_client[conv_id]

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
                        self.item_hypergraph.append(meta_item_hypergraph)
                for entity in dialog["entity"]:
                    if entity not in conv_entity_list:
                        entity_hypergraph_id_to_client[self.entity_hypergraph_id] = client_id
                        conv_entity_list.add(entity)
                        meta_entity_hypergraph.add(entity)
                        self.entity_popularity[entity] = self.entity_popularity.get(entity, 0) + 1
                        self.entity_hypergraph_id += 1
                        self.entity_hypergraph.append(meta_entity_hypergraph)
                for word in dialog["word"]:
                    if word not in conv_word_list:
                        word_hypergraph_id_to_client[self.word_hypergraph_id] = client_id
                        conv_word_list.add(word)
                        meta_word_hypergraph.add(word)
                        self.word_popularity[word] = self.word_popularity.get(word, 0) + 1
                        self.word_hypergraph_id += 1
                        self.word_hypergraph.append(meta_word_hypergraph)

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

        def func(hypergraph: List[List[int]], popularity: Dict[int, int]) -> List[float]:
            get_pop = popularity.get
            return [
                sum(get_pop(item, 0) for item in meta) / len(meta) if meta else 0.0
                for meta in hypergraph
            ]

        self.item_hypergraph_popularity = func(self.item_hypergraph, self.item_popularity)
        self.entity_hypergraph_popularity = func(self.entity_hypergraph, self.entity_popularity)
        self.word_hypergraph_popularity = func(self.word_hypergraph, self.word_popularity)
        return (self.item_hypergraph_popularity, self.entity_hypergraph_popularity, self.word_hypergraph_popularity)