import os
from typing import Dict

class FedUnlConfig:
    def __init__(self, args:Dict) -> None:
        self.task = args["task"]

        self.model_name = args["model_name"]
        self.dataset_name = args["dataset_name"]

        self.partition_methon = args["partition_methon"]
        self.partition_mode = args["partition_mode"]

        self.unlearning_layer = args["unlearning_layer"]
        self.unlearning_sample_methon = args["unlearning_sample_methon"]

        self.n_client_per_proc = args["n_client_per_proc"]
        self.n_proc = args["n_proc"]
        self.n_client = args["n_client"]

        self.aggregate_methon = args["aggregate_methon"]
        self.aggregate_rate = args["aggregate_rate"]

        self.epochs = args["epochs"]
        self.batch_size = args["batch_size"]
        self.learning_rate = args["learning_rate"]

        self.load_path = os.path.join(
            "save", "partition",
            f"{self.dataset_name}_{self.partition_methon}_{self.n_client}"
        )
        self.save_path = os.path.join(
            "save", "unlearning",
            f"{self.model_name}_{self.unlearning_sample_methon}_{self.dataset_name}_{self.partition_methon}_{self.n_client}"
        )

        self.load_random_config(args)
        self.load_model_config(args)

        return
    
    def load_random_config(self, args:Dict) -> None:
        random_config = args["random_config"]
        return
    
    def load_model_config(self, args:Dict) -> None:
        model_config = args["model_config"]

        self.mlp_config = model_config["mlp"]

        self.hycorec_config = model_config["hycorec"]
        return