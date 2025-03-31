import os
import json
import random

from .config import PartitionConfig
from fedunlcrs.utils import FedUnlDataLoader

class PartitionWorker:
    def __init__(self, config:PartitionConfig) -> None:
        os.makedirs(config.save_dir, exist_ok=True)
        self.config = config
        self.dataloader = FedUnlDataLoader(self.config.dataset_name, 1, None, None)

        if self.config.partition_methon == "random":
            self.methon_random()

        return
    
    def methon_random(self) -> None:
        user_id_list = set()
        conv_id_list = set()
        item_id_list = list(range(self.dataloader.n_item))
        entity_id_list = list(range(self.dataloader.n_entity))
        word_id_list = list(range(self.dataloader.n_word))

        for conv in self.dataloader.train_dataset:
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