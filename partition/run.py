import os
import json
import numpy as np
from k_means_constrained import KMeansConstrained

from utils import get_dataset, get_edger
from partition.pretrain import run_pretrain

def run_partition(dataset, partition_model):
    train_dataset, valid_dataset, test_dataset = get_dataset(dataset)
    print(f"[+] Load dataset with size of {len(train_dataset)} {len(valid_dataset)} {len(test_dataset)}")
    item_edger, entity_edger, word_edger = get_edger(dataset)
    print(f"[+] Load edger with size of {len(item_edger)} {len(entity_edger)} {len(word_edger)}")

    if not os.path.isfile(os.path.join("save", "pretrain", f"{dataset}-{partition_model}", ".built")):
        item2idx = json.load(open(os.path.join("data", dataset, "entity2id.json"), "r", encoding="utf-8"))
        entity2idx = json.load(open(os.path.join("data", dataset, "entity2id.json"), "r", encoding="utf-8"))
        word2idx = json.load(open(os.path.join("data", dataset, "token2id.json"), "r", encoding="utf-8"))
        item_embedding, entity_embedding, word_embedding, dialog_embedding = run_pretrain(
            dataset, partition_model, train_dataset,
            item_edger, entity_edger, word_edger,
            item2idx, entity2idx, word2idx
        )
        for path in ["save", "save/pretrain", f"save/pretrain/{dataset}-{partition_model}"]:
            if os.path.isdir(os.path.join(path)):
                continue
            os.mkdir(path)
        np.save(open(os.path.join("save", "pretrain", f"{dataset}-{partition_model}", "item_pretrain.npy"), "wb"), item_embedding)
        np.save(open(os.path.join("save", "pretrain", f"{dataset}-{partition_model}", "entity_pretrain.npy"), "wb"), entity_embedding)
        np.save(open(os.path.join("save", "pretrain", f"{dataset}-{partition_model}", "word_pretrain.npy"), "wb"), word_embedding)
        np.save(open(os.path.join("save", "pretrain", f"{dataset}-{partition_model}", "dialog_pretrain.npy"), "wb"), dialog_embedding)
        print(f"[+] Save embedding in {os.path.join('save', 'pretrain', f'{dataset}-{partition_model}')}")
        with open(os.path.join("save", "pretrain", f"{dataset}-{partition_model}", ".built"), "w") as built_file:
            built_file.write("\n")
    print(f"[+] Finish embedding pretrain!")

    item_embedding = np.load(open(os.path.join("save", "pretrain", f"{dataset}-{partition_model}", "item_pretrain.npy"), "rb"))
    entity_embedding = np.load(open(os.path.join("save", "pretrain", f"{dataset}-{partition_model}", "entity_pretrain.npy"), "rb"))
    word_embedding = np.load(open(os.path.join("save", "pretrain", f"{dataset}-{partition_model}", "word_pretrain.npy"), "rb"))
    dialog_embedding = np.load(open(os.path.join("save", "pretrain", f"{dataset}-{partition_model}", "dialog_pretrain.npy"), "rb"))

    n_clusters = 8
    if not os.path.isfile(os.path.join("save", "partition", f"{dataset}-{n_clusters}", ".built")):
        item_split_label = split_embedding(item_embedding, n_clusters)
        print(f"[+] Split item embedding into {n_clusters} clusters")
        entity_split_label = split_embedding(entity_embedding, n_clusters)
        print(f"[+] Split entity embedding into {n_clusters} clusters")
        word_split_label = split_embedding(word_embedding, n_clusters)
        print(f"[+] Split word embedding into {n_clusters} clusters")
        dialog_split_label = split_embedding(dialog_embedding, n_clusters)
        print(f"[+] Split dialog embedding into {n_clusters} clusters")

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
        print(f"[+] Save {n_clusters} sub dataset in {os.path.join('save', 'partition', f'{dataset}-{n_clusters}')}")
        with open(os.path.join("save", "partition", f"{dataset}-{n_clusters}", ".built"), "w") as built_file:
            built_file.write("\n")
    print(f"[+] Finish dataset partition!")
        
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