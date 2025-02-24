import argparse

from fedunlcrs.pretrain import run_pretrain
from fedunlcrs.partition import run_partition

dataset = "opendialkg"
partition_model = "mlp"

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--task", type=str)
    args = args.parse_args()

    task = args.task

    if task in ["pretrain", "all"]:
        run_pretrain(dataset, partition_model)
    if task in ["partition", "all"]:
        run_partition(dataset, partition_model)