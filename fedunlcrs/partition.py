import os
import json
import numpy as np
from loguru import logger
from time import perf_counter
from typing import Dict, List
from k_means_constrained import KMeansConstrained

def run_partition(task_config:Dict) -> None:
    pretrain_dir :str = task_config["pretrain_dir"]
    save_dir     :str = task_config["save_dir"]
    n_clusters   :int = task_config["n_clusters"]

    item_embedding   = np.load(open(os.path.join(pretrain_dir, "item_pretrain.npy"), "rb"))
    logger.info(f"Load item   pretrain embedding from {os.path.join(pretrain_dir, "item_pretrain.npy")}")
    entity_embedding = np.load(open(os.path.join(pretrain_dir, "entity_pretrain.npy"), "rb"))
    logger.info(f"Load entity pretrain embedding from {os.path.join(pretrain_dir, "entity_pretrain.npy")}")
    word_embedding   = np.load(open(os.path.join(pretrain_dir, "word_pretrain.npy"), "rb"))
    logger.info(f"Load word   pretrain embedding from {os.path.join(pretrain_dir, "word_pretrain.npy")}")
    dialog_embedding = np.load(open(os.path.join(pretrain_dir, "dialog_pretrain.npy"), "rb"))
    logger.info(f"Load dialog pretrain embedding from {os.path.join(pretrain_dir, "dialog_pretrain.npy")}")

    start_time = perf_counter()
    item_split_label   = split_embedding(item_embedding, n_clusters)
    logger.info(f"Split item   embedding into {n_clusters} parts ({perf_counter() - start_time:.2f}s)")
    start_time = perf_counter()
    entity_split_label = split_embedding(entity_embedding, n_clusters)
    logger.info(f"Split entity embedding into {n_clusters} parts ({perf_counter() - start_time:.2f}s)")
    start_time = perf_counter()
    word_split_label   = split_embedding(word_embedding, n_clusters)
    logger.info(f"Split word   embedding into {n_clusters} parts ({perf_counter() - start_time:.2f}s)")
    start_time = perf_counter()
    dialog_split_label = split_embedding(dialog_embedding, n_clusters)
    logger.info(f"Split dialog embedding into {n_clusters} parts ({perf_counter() - start_time:.2f}s)")
    start_time = perf_counter()

    os.makedirs(save_dir, exist_ok=True)
    for idx in range(n_clusters):
        sub_dataset_mask = {
            "item_mask": item_split_label[idx],
            "entity_mask": entity_split_label[idx],
            "word_mask": word_split_label[idx],
            "dialog_mask": dialog_split_label[idx],
        }
        json.dump(
            sub_dataset_mask,
            open(
                os.path.join(save_dir, f"sub_dataset_mask_{idx + 1}_{n_clusters}.json"),
                "w", encoding="utf-8"
            ),
            indent=4
        )
        logger.info(f"Save [{idx + 1}/{n_clusters}] sub dataset file in {os.path.join(save_dir, f"sub_dataset_mask_{idx + 1}_{n_clusters}.json")}")
        
    return

def split_embedding(embedding:np.ndarray, n_clusters:int) -> List:
    kmeans = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=int(0.8 * embedding.shape[0] / n_clusters),
        size_max=int(1.2 * embedding.shape[0] / n_clusters),
        max_iter=32,
        n_jobs=-1,
    )
    labels = kmeans.fit_predict(embedding).tolist()
    res = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        res[label].append(idx)
    return res