import os
import json
import torch
import numpy as np

from utils import get_dataset, get_edger
from partition.pretrain import run_pretrain

def run_partition(dataset, partition_model):
    train_dataset, valid_dataset, test_dataset = get_dataset(dataset)
    print(f"[+] Get dataset with size of {len(train_dataset)} {len(valid_dataset)} {len(test_dataset)}.")
    item_edger, entity_edger, word_edger = get_edger(dataset)
    print(f"[+] Get edger with size of {len(item_edger)} {len(entity_edger)} {len(word_edger)}.")

    if not os.path.isfile(os.path.join("save", "pretrain", dataset, ".built")):
        item2idx = json.load(open(os.path.join("data", dataset, "entity2id.json"), "r", encoding="utf-8"))
        entity2idx = json.load(open(os.path.join("data", dataset, "entity2id.json"), "r", encoding="utf-8"))
        word2idx = json.load(open(os.path.join("data", dataset, "token2id.json"), "r", encoding="utf-8"))
        item_embedding, entity_embedding, word_embedding = run_pretrain(
            dataset, partition_model, train_dataset,
            item_edger, entity_edger, word_edger,
            item2idx, entity2idx, word2idx
        )
        for path in ["save", "save/pretrain", f"save/pretrain/{dataset}"]:
            if os.path.isdir(os.path.join(path)):
                continue
            os.mkdir(path)
        np.save(open(os.path.join("save", "pretrain", dataset, "item_pretrain.npy"), "wb"), item_embedding)
        np.save(open(os.path.join("save", "pretrain", dataset, "entity_pretrain.npy"), "wb"), entity_embedding)
        np.save(open(os.path.join("save", "pretrain", dataset, "word_pretrain.npy"), "wb"), word_embedding)
        with open(os.path.join("save", "pretrain", dataset, ".built"), "w") as built_file:
            built_file.write("\n")

    item_embedding = np.load(open(os.path.join("save", "pretrain", dataset, "item_pretrain.npy"), "rb"))
    entity_embedding = np.load(open(os.path.join("save", "pretrain", dataset, "entity_pretrain.npy"), "rb"))
    word_embedding = np.load(open(os.path.join("save", "pretrain", dataset, "word_pretrain.npy"), "rb"))

    print(item_embedding.shape, entity_embedding.shape, word_embedding.shape)

    return