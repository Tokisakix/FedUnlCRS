import yaml
import argparse

from fedunlcrs.partition import PartitionConfig, PartitionWorker

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, help="the config file path")
    args = args.parse_args()

    config = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    if config["task"] == "partition":
        partition_config = PartitionConfig(config)
        PartitionWorker(partition_config)

    exit(0)