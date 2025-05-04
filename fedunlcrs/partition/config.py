import os
from typing import Dict

class PartitionConfig:
    def __init__(self, args:Dict) -> None:
        self.task = args["task"]
        self.dataset_name = args["dataset_name"]

        self.partition_methon = args["partition_methon"]
        self.partition_num = args["partition_num"]
        self.save_dir = os.path.join("/gz-data/save", self.task, f"{self.dataset_name}_{self.partition_methon}_{self.partition_num}")

        self.load_random_config(args)

        return
    
    def load_random_config(self, args:Dict) -> None:
        random_config = args["random_config"]
        return