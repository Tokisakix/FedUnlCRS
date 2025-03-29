import os
import json
import yaml
import argparse
from loguru import logger

from fedunlcrs.pretrain import run_pretrain
from fedunlcrs.partition import run_partition
from fedunlcrs.federated import run_federated
from fedunlcrs.unlearning import run_unlearning

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--task_config", type=str, default=None)
    args.add_argument("--model_config", type=str, default=None)
    args = args.parse_args()

    task_config  = yaml.safe_load(open(args.task_config, "r", encoding="utf-8"))
    save_path = task_config["save_dir"]
    os.makedirs(save_path, exist_ok=True)
    logger.add(os.path.join(save_path, "log.txt"), level="INFO", mode="w")
    logger.info(f"Get the task config file:\n{json.dumps(task_config, indent=4)}")
    model_config = yaml.safe_load(open(args.model_config, "r", encoding="utf-8"))
    logger.info(f"Get the model config file:\n{json.dumps(model_config, indent=4)}")

    match task_config["task"]:
        case "pretrain":
            run_pretrain(task_config, model_config)
        case "partition":
            run_partition(task_config)
        case "federated":
            run_federated(task_config, model_config)
        case "unlearning":
            run_unlearning(task_config, model_config)

    exit(0)