import yaml
import argparse

from fedunlcrs.partition import PartitionConfig, PartitionWorker
from fedunlcrs.unlearning import FedUnlConfig, FedUnlWorker

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, help="the config file path")
    args = args.parse_args()

    config = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    if config["task"] == "partition":
        PartitionWorker(PartitionConfig(config))
    elif config["task"] == "unlearning":
        FedUnlWorker(FedUnlConfig(config))

    exit(0)