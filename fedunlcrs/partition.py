import os
import json
import numpy as np
from loguru import logger
from time import perf_counter
from typing import Dict, List
from k_means_constrained import KMeansConstrained
from tqdm import tqdm

from fedunlcrs.utils import get_dataset

def run_partition(task_config:Dict) -> None:
    dataset      :str = task_config["dataset"]
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
        
    idx_to_client = build_idx_to_client(dataset, n_clusters, item_split_label, entity_split_label, word_split_label, dialog_split_label)
    json.dump(
        idx_to_client,
        open(
            os.path.join(save_dir, f"idx_to_client.json"),
            "w", encoding="utf-8"
        ),
    )
    logger.info(f"Save idx to client file in {os.path.join(save_dir, f"idx_to_client.json")}")

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

def build_idx_to_client(
        dataset:str, n_client:int,
        item_split_label:List, entity_split_label:List, word_split_label:List,
        dialog_split_label:List) -> Dict:
    
    train_dataset, _, _ = get_dataset(dataset)
    idx_to_client = {}
    item_hyper_graph_id = 0
    entity_hyper_graph_id = 0
    word_hyper_graph_id = 0

    idx_to_client["item_to_client"]   = {item:client   for client in range(n_client) for item   in item_split_label[client]}
    idx_to_client["entity_to_client"] = {entity:client for client in range(n_client) for entity in entity_split_label[client]}
    idx_to_client["word_to_client"]   = {word:client   for client in range(n_client) for word   in word_split_label[client]}
    idx_to_client["dialog_to_client"] = {dialog:client for client in range(n_client) for dialog in dialog_split_label[client]}
    idx_to_client["user_to_client"]        = {}
    idx_to_client["item_hypg_to_client"]   = {}
    idx_to_client["entity_hypg_to_client"] = {}
    idx_to_client["word_hypg_to_client"]   = {}

    max_user_id = 0
    max_dialog_id = 0
    for conv in tqdm(train_dataset):
        user_id   = int(conv["conv_id"]) if conv["user_id"] is None else int(conv["user_id"])
        dialog_id = int(conv["conv_id"])
        conv_item_list   = []
        conv_entity_list = []
        conv_word_list   = []

        for dialog in conv["dialogs"]:
            conv_item_list += dialog["item"]
            conv_entity_list += dialog["entity"]
            conv_word_list += dialog["word"]

        conv_item_list = list(set(conv_item_list))
        conv_entity_list = list(set(conv_entity_list))
        conv_word_list = list(set(conv_word_list))

        dialog_client_id = idx_to_client["dialog_to_client"][dialog_id]
        idx_to_client["user_to_client"][user_id] = dialog_client_id

        for _ in conv_item_list:
            idx_to_client["item_hypg_to_client"][item_hyper_graph_id] = dialog_client_id
            item_hyper_graph_id += 1
        for _ in conv_entity_list:
            idx_to_client["entity_hypg_to_client"][entity_hyper_graph_id] = dialog_client_id
            entity_hyper_graph_id += 1
        for _ in conv_word_list:
            idx_to_client["word_hypg_to_client"][word_hyper_graph_id] = dialog_client_id
            word_hyper_graph_id += 1
        max_user_id = max(max_user_id, user_id)
        max_dialog_id = max(max_dialog_id, dialog_id)

    # print(item_hyper_graph_id, entity_hyper_graph_id, word_hyper_graph_id, max_user_id, max_dialog_id)

    return idx_to_client