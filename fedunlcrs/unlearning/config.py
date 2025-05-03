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
        self.topk = args["topk"]

        self.ablation_layer = args["ablation_layer"]

        self.n_client_per_proc = args["n_client_per_proc"]
        self.n_proc = args["n_proc"]
        self.n_client = args["n_client"]

        self.aggregate_methon = args["aggregate_methon"]
        self.aggregate_rate = args["aggregate_rate"]

        self.epochs = args["epochs"]
        self.batch_size = args["batch_size"]
        self.learning_rate = args["learning_rate"]
        self.emb_dim = args["model_config"]["hycorec"]["emb_dim"]

        self.load_path = os.path.join(
            "save", "partition",
            f"{self.dataset_name}_{self.partition_methon}_{self.n_client}"
        )
        self.save_path = os.path.join(
            "save", "unlearning",
            f"{self.model_name}_{self.dataset_name}_{self.unlearning_sample_methon}_{self.ablation_layer}_{self.n_client}_{self.emb_dim}_{self.aggregate_rate}"
        )
        self.evaluate_path = os.path.join(
            "save", "evaluate",
            f"{self.model_name}_{self.dataset_name}_{self.unlearning_sample_methon}_{self.ablation_layer}_{self.n_client}_{self.emb_dim}_{self.aggregate_rate}"
        )

        self.load_model_config(args)

        return
    
    def load_model_config(self, args:Dict) -> None:
        model_config = args["model_config"]
        self.mlp_config = model_config["mlp"]
        self.hycorec_config = model_config["hycorec"]
        self.kbrd_config = model_config["kbrd"]
        self.bert_config = model_config["bert"]
        self.gru4rec_config = model_config["gru4rec"]
        self.kgsf_config = model_config["kgsf"]
        self.ntrd_config = model_config["ntrd"]
        self.redial_config = model_config["redial"]
        self.sasrec_config = model_config["sasrec"]
        self.textcnn_config = model_config["textcnn"]
        self.tgredial_config = model_config["tgredial"]
        self.mhim_config = model_config["mhim"]
        return