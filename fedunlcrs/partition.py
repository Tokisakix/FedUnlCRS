import os
import json
import numpy as np
from typing import Dict, List
from k_means_constrained import KMeansConstrained

def run_partition(task_config:Dict) -> None:
    pretrain_dir :str = task_config["pretrain_dir"]
    save_dir     :str = task_config["save_dir"]
    n_clusters   :int = task_config["n_clusters"]

    item_embedding = np.load(open(os.path.join(pretrain_dir, "item_pretrain.npy"), "rb"))
    entity_embedding = np.load(open(os.path.join(pretrain_dir, "entity_pretrain.npy"), "rb"))
    word_embedding = np.load(open(os.path.join(pretrain_dir, "word_pretrain.npy"), "rb"))
    dialog_embedding = np.load(open(os.path.join(pretrain_dir, "dialog_pretrain.npy"), "rb"))

    item_split_label = split_embedding(item_embedding, n_clusters)
    entity_split_label = split_embedding(entity_embedding, n_clusters)
    word_split_label = split_embedding(word_embedding, n_clusters)
    dialog_split_label = split_embedding(dialog_embedding, n_clusters)

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
        
    return

def split_embedding(embedding:np.ndarray, n_clusters:int) -> List:
    kmeans = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=int(0.8 * embedding.shape[0] / n_clusters),
        size_max=int(1.2 * embedding.shape[0] / n_clusters),
        max_iter=8,
        n_jobs=4,
    )
    labels = kmeans.fit_predict(embedding).tolist()
    res = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        res[label].append(idx)
    return res