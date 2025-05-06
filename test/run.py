import argparse
import subprocess

PYTHON = R"/usr/local/anaconda3/bin/python"
PARTITION_CONFIG  = R"config/partition_config.yaml"
UNLEARNING_CONFIG = R"config/unlearning_config.yaml"

MODELS = ["mlp", "hycorec", "kbrd", "bert", "gru4rec", "kgsf", "ntrd", "redial", "sasrec", "textcnn", "tgredial", "mhim", "rec"]
DATASETS = ["opendialkg", "durecdial", "hredial", "htgredial"]

REC_METRICS = ["R@10", "R@50", "M@10", "M@50", "N@10", "N@50"]
COV_METRICS = ["Dist-2", "Dist-3", "Dist-4", "BLEU-2", "BLEU-3", "BLEU-4"]
FAIRENESS_METRICS = ["APR@5", "APR@10", "APR@15", "APR@20", "LTR@5", "LTR@10", "LTR@15", "LTR@20", "Cov@5", "Cov@10", "Cov@15", "Cov@20", "Gini@5", "Gini@10", "Gini@15", "Gini@20", "KL@5", "KL@10", "KL@15", "KL@20", "Diff@5", "Diff@10", "Diff@15", "Diff@20"]

METHONS = ["random", "topk"]
ABLATION_LAYERS = ["item", "entity", "word"]
UNLEARNING_LAYERS = ["user", "conv", "item", "entity", "word", "item_hypergraph", "entity_hypergraph", "word_hypergraph"]

N_CLIENTS = [4, 8, 16, 32]
EMBEDDING_DIM = [32, 64, 128, 256]
AGGREGATE_RATE = [0.005, 0.001, 0.0005, 0.0001]

def run_cmd(command:str):
    process = subprocess.Popen(
        command.split(" "),
    )
    process.communicate()
    return

def run_all_partition():
    for dataset in DATASETS:
        for n_client in N_CLIENTS:
            command = f"{PYTHON} main.py --config {PARTITION_CONFIG} --dataset {dataset} --n_client {n_client}"
            print("[SCRIPT]", command)
            run_cmd(command)
    return

def run_all_model():
    for model in MODELS:
        for dataset in DATASETS:
            command = f"{PYTHON} main.py --config {UNLEARNING_CONFIG} --model {model} --dataset {dataset}"
            print("[SCRIPT]", command)
            run_cmd(command)
    return

def run_ablation():
    model = "hycorec"
    for dataset in DATASETS:
        for ablation in ABLATION_LAYERS:
            command = f"{PYTHON} main.py --config {UNLEARNING_CONFIG} --model {model} --dataset {dataset} --ablation {ablation}"
            print("[SCRIPT]", command)
            run_cmd(command)

def run_n_client():
    model = "hycorec"
    for dataset in DATASETS:
        for n_client in N_CLIENTS:
            command = f"{PYTHON} main.py --config {UNLEARNING_CONFIG} --model {model} --dataset {dataset} --n_client {n_client}"
            print("[SCRIPT]", command)
            run_cmd(command)

def run_embedding_dim():
    model = "hycorec"
    for dataset in DATASETS:
        for embedding_dim in EMBEDDING_DIM:
            command = f"{PYTHON} main.py --config {UNLEARNING_CONFIG} --model {model} --dataset {dataset} --embedding_dim {embedding_dim}"
            print("[SCRIPT]", command)
            run_cmd(command)

def run_aggregate_rate():
    model = "hycorec"
    for dataset in DATASETS:
        for aggregate_rate in AGGREGATE_RATE:
            command = f"{PYTHON} main.py --config {UNLEARNING_CONFIG} --model {model} --dataset {dataset} --aggregate_rate {aggregate_rate}"
            print("[SCRIPT]", command)
            run_cmd(command)
    return

def run_unlearning():
    model = "hycorec"
    for dataset in DATASETS:
        command = f"{PYTHON} main.py --config {UNLEARNING_CONFIG} --model {model} --dataset {dataset} --unlearning True"
        print("[SCRIPT]", command)
        run_cmd(command)
    return

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--task", type=int)
    args = args.parse_args()

    if args.task == 0:
        run_all_partition()
    elif args.task == 1:
        run_all_model()
    elif args.task == 2:
        run_ablation()
    elif args.task == 3:
        run_n_client()
    elif args.task == 4:
        run_embedding_dim()
    elif args.task == 5:
        run_aggregate_rate()
    elif args.task == 6:
        run_unlearning()
    exit(0)