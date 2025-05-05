import yaml
import argparse

from fedunlcrs.partition import PartitionConfig, PartitionWorker
from fedunlcrs.unlearning import FedUnlConfig, FedUnlWorker

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, help="the config file path")
    args.add_argument("--model", type=str, help="model name", default=None)
    args.add_argument("--dataset", type=str, help="dataset name", default=None)
    args.add_argument("--methon", type=str, help="unlearning methon", default=None)
    args.add_argument("--ablation", type=str, help="ablation study", default=None)
    args.add_argument("--n_client", type=int, help="n client", default=None)
    args.add_argument("--embedding_dim", type=int, help="embedding dim", default=None)
    args.add_argument("--aggregate_rate", type=float, help="aggregate rate", default=None)
    args = args.parse_args()

    config = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    if config["task"] == "partition":
        if args.dataset:
            config["dataset_name"] = args.dataset
        if args.n_client:
            config["partition_num"] = args.n_client
        config = PartitionConfig(config)
        PartitionWorker(config)
    elif config["task"] == "unlearning":
        if args.model:
            config["model_name"] = args.model
        if args.dataset:
            config["dataset_name"] = args.dataset
        if args.methon:
            config["unlearning_sample_methon"] = args.methon
        config["model_config"]["hycorec"]["ablation_layer"] = args.ablation
        if args.n_client:
            config["n_client"] = args.n_client
        if args.embedding_dim:
            config["emb_dim"] = args.embedding_dim
        if args.aggregate_rate:
            config["aggregate_rate"] = args.aggregate_rate
        config = FedUnlConfig(config)
        FedUnlWorker(config)

    exit(0)