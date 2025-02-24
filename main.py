import yaml
import argparse

from fedunlcrs.pretrain import run_pretrain
from fedunlcrs.partition import run_partition

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--task_config", type=str, default=None)
    args.add_argument("--model_config", type=str, default=None)
    args = args.parse_args()

    task_config  = yaml.safe_load(open(args.task_config, "r", encoding="utf-8"))
    model_config = yaml.safe_load(open(args.model_config, "r", encoding="utf-8"))

    match task_config["task"]:
        case "pretrain":
            run_pretrain(task_config, model_config)
        case "partition":
            run_partition(task_config)

    exit(0)