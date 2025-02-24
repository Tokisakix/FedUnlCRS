import os
import json
import numpy as np
from k_means_constrained import KMeansConstrained

def run_partition(dataset, partition_model):
    item_embedding = np.load(open(os.path.join("save", "pretrain", f"{dataset}-{partition_model}", "item_pretrain.npy"), "rb"))
    entity_embedding = np.load(open(os.path.join("save", "pretrain", f"{dataset}-{partition_model}", "entity_pretrain.npy"), "rb"))
    word_embedding = np.load(open(os.path.join("save", "pretrain", f"{dataset}-{partition_model}", "word_pretrain.npy"), "rb"))
    dialog_embedding = np.load(open(os.path.join("save", "pretrain", f"{dataset}-{partition_model}", "dialog_pretrain.npy"), "rb"))

    n_clusters = 8
    if not os.path.isfile(os.path.join("save", "partition", f"{dataset}-{n_clusters}", ".built")):
        item_split_label = split_embedding(item_embedding, n_clusters)
        entity_split_label = split_embedding(entity_embedding, n_clusters)
        word_split_label = split_embedding(word_embedding, n_clusters)
        dialog_split_label = split_embedding(dialog_embedding, n_clusters)

        for path in ["save", "save/partition", f"save/partition/{dataset}-{n_clusters}"]:
                if os.path.isdir(os.path.join(path)):
                    continue
                os.mkdir(path)
        for idx in range(n_clusters):
            sub_dataset_mask = {
                "item_mask": item_split_label[idx],
                "entity_mask": entity_split_label[idx],
                "word_mask": word_split_label[idx],
                "dialog_mask": dialog_split_label[idx],
            }
            json.dump(sub_dataset_mask, open(os.path.join("save", "partition", f"{dataset}-{n_clusters}", f"sub_dataset_mask_{idx + 1}_{n_clusters}.json"), "w", encoding="utf-8"), indent=4)
        with open(os.path.join("save", "partition", f"{dataset}-{n_clusters}", ".built"), "w") as built_file:
            built_file.write("\n")
        
    return

def split_embedding(embedding, n_clusters):
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