import os
import json
from typing import Dict, List

from fedunlcrs.utils import get_dataset, get_edger

class FedUnlDataLoader:
    def __init__(self, dataset_name:str, batch_size:int, partition_mask:Dict=None, parition_mode:str=None) -> None:
        # init variable
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.partition_mask = partition_mask
        self.parition_mode = parition_mode

        # load raw data
        (raw_train_dataset, raw_valid_dataset, raw_test_dataset), (raw_item_edger, raw_entity_edger, raw_word_edger) = self.load_raw_data()

        # build edger
        self.item_edger = self.build_edger(raw_item_edger, self.item2id, self.n_item)
        self.entity_edger = self.build_edger(raw_entity_edger, self.entity2id, self.n_entity)
        self.word_edger = self.build_edger(raw_word_edger, self.word2id, self.n_word)

        # build dataset
        self.train_dataset = self.build_dataset(raw_train_dataset)
        self.valid_dataset = self.build_dataset(raw_valid_dataset)
        self.test_dataset = self.build_dataset(raw_test_dataset)

        return
    
    def load_raw_data(self) -> None:
        raw_train_dataset, raw_valid_dataset, raw_test_dataset = get_dataset(self.dataset_name)
        raw_item_edger, raw_entity_edger, raw_word_edger = get_edger(self.dataset_name)
        self.item2id = json.load(open(os.path.join("data", self.dataset_name, "entity2id.json"), "r", encoding="utf-8"))
        self.entity2id = json.load(open(os.path.join("data", self.dataset_name, "entity2id.json"), "r", encoding="utf-8"))
        self.word2id = json.load(open(os.path.join("data", self.dataset_name, "token2id.json"), "r", encoding="utf-8"))

        self.n_item = len(self.item2id) + 1
        self.n_entity = len(self.entity2id) + 1
        self.n_word = len(self.word2id)
        return (raw_train_dataset, raw_valid_dataset, raw_test_dataset), (raw_item_edger, raw_entity_edger, raw_word_edger)
    
    def build_edger(self, raw_edger:Dict[str, str], str2id_dict:Dict, n:int) -> List[List[int]]:
        edger = []
        id2str_dict = {str2id_dict[_str]:_str for _str in str2id_dict}

        for id_a in range(n):
            id_list = []
            if id_a in id2str_dict:
                str_a = id2str_dict[id_a]
                str_list = raw_edger.get(str_a, [])
                for str_b in str_list:
                    if str_b in str2id_dict:
                        id_b = str2id_dict[str_b]
                        id_list.append(id_b)
            edger.append(id_list)

        assert len(edger) == n
        return edger
    
    def build_dataset(self, raw_dataset:List[Dict], use_mask:bool=True) -> List[Dict]:
        dataset = []

        for conv in raw_dataset:
            user_id = int(conv["user_id"])
            if use_mask and self.parition_mode == "user" and user_id not in self.partition_mask["user_mask"]:
                continue

            conv_id = int(conv["conv_id"])
            if use_mask and self.parition_mode == "conv" and conv_id not in self.partition_mask["conv_mask"]:
                continue

            conv_item_list = set()
            conv_entity_list = set()
            conv_word_list = set()

            for dialog in conv["dialogs"]:
                role = dialog["role"]
                labels = dialog["item"]
                
                if role == "Recommender":
                    for label in labels:
                        if label in self.item2id:
                            label = self.item2id[label]
                            meta_data = {
                                "user_id": user_id,
                                "conv_id": conv_id,
                                "item": list(conv_item_list),
                                "entity": list(conv_entity_list),
                                "word": list(conv_word_list),
                                "label": label,
                            }
                            dataset.append(meta_data)

                for item in dialog["item"]:
                    if item in self.item2id:
                        item_id = self.item2id[item]
                        if use_mask and self.parition_mode == "item" and item_id not in self.partition_mask["item_mask"]:
                            continue
                        conv_item_list.add(item_id)
                for entity in dialog["entity"]:
                    if entity in self.entity2id:
                        entity_id = self.entity2id[entity]
                        if use_mask and self.parition_mode == "entity" and entity_id not in self.partition_mask["entity_mask"]:
                            continue
                        conv_entity_list.add(entity_id)
                for word in dialog["word"]:
                    if word in self.word2id:
                        word_id = self.word2id[word]
                        if use_mask and self.parition_mode == "word" and word_id not in self.partition_mask["word_mask"]:
                            continue
                        conv_word_list.add(word_id)

        return dataset

    def get_fed_data(self, mode:str, batch_size:int) -> List[Dict]:
        batch_data = []

        if mode == "train":
            dataset = self.train_dataset
        elif mode == "valid":
            dataset = self.valid_dataset
        elif mode == "test":
            dataset = self.test_dataset

        batch = []
        for meta_data in dataset:
            batch.append(meta_data)
            if len(batch) >= batch_size:
                batch_data.append(batch)
                batch = []
        if len(batch) > 0:
            batch_data.append(batch)

        return batch_data

    def get_data(self, mode:str, batch_size:int) -> List[Dict]:
        batch_data = []

        if mode == "train":
            dataset = self.train_dataset
        elif mode == "valid":
            dataset = self.valid_dataset
        elif mode == "test":
            dataset = self.test_dataset

        batch = []
        for meta_data in dataset:
            batch.append(meta_data)
            if len(batch) >= batch_size:
                batch_data.append(batch)
                batch = []
        if len(batch) > 0:
            batch_data.append(batch)

        return batch_data