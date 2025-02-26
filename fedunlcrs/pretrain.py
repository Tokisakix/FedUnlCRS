import os
import json
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from typing import Dict, List, Tuple

from fedunlcrs.utils import get_dataloader, get_dataset, get_edger
from fedunlcrs.model import get_classifer, PretrainEmbeddingModel

def train_pretrain(
        task_config:Dict, model_config:Dict, train_dataset:List,
        item_edger:Dict, entity_edger:Dict, word_edger:Dict,
        item2idx:Dict, entity2idx:Dict, word2idx:Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataloader = get_dataloader(train_dataset, item2idx, entity2idx, word2idx)
    logger.info(f"Build dataloader with size of {len(dataloader)}")

    pretrain_model:str = model_config["model"]
    embedding_dim :int = model_config["embedding_dim"]
    n_item        :int = len(item2idx) + 1
    n_entity      :int = len(entity2idx) + 1
    n_word        :int = len(word2idx)
    device        :str = task_config["device"]
    epochs        :int = task_config["epochs"]
    learning_rate :float = float(task_config["learning_rate"])

    classifer = get_classifer(pretrain_model)(embedding_dim, n_item)
    model = PretrainEmbeddingModel(n_item, n_entity, n_word, embedding_dim, classifer, device).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger.info(f"Build Model:\n{model}")

    logger.info("Start pretraining")
    for epoch in range(1, epochs + 1, 1):
        tot_loss = 0.0

        with tqdm(total= len(dataloader)) as pretrain_tqdm:
            pretrain_tqdm.set_description(f"Epoch: {epoch}/{epochs}")
            for meta_data in dataloader:
                item_list = torch.LongTensor(meta_data["item"]).to(device)
                entity_list = torch.LongTensor(meta_data["entity"]).to(device)
                word_list = torch.LongTensor(meta_data["word"]).to(device)
                label = torch.LongTensor(meta_data["label"]).to(device)

                output = model(item_list, entity_list, word_list, item_edger, entity_edger, word_edger)
                optimizer.zero_grad()
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                loss = loss.cpu().item()
                pretrain_tqdm.set_postfix(loss=f"{loss:.6f}")
                pretrain_tqdm.update(1)
                tot_loss += loss

        logger.info(f"[Epoch:{epoch}/{epochs} Loss:{tot_loss/len(dataloader):.6f}]")
    logger.info("Finish pretraining")

    item_embedding = model.item_embedding.weight.detach().cpu().numpy()
    logger.info("Get item    embedding")
    entity_embedding = model.entity_embedding.weight.detach().cpu().numpy()
    logger.info("Get entity  embedding")
    word_embedding = model.word_embedding.weight.detach().cpu().numpy()
    logger.info("Get word    embedding")

    dialog_embedding = np.zeros((len(train_dataset), embedding_dim), dtype=np.float32)
    for idx, conv in enumerate(tqdm(train_dataset)):
        dialog_item_list = []
        dialog_entity_list = []
        dialog_word_list = []
        for dialog in conv["dialogs"]:
            dialog_item_list += dialog["item"]
            dialog_item_list += dialog["entity"]
            dialog_item_list += dialog["word"]
        dialog_item_list = set(dialog_item_list)
        dialog_entity_list = set(dialog_entity_list)
        dialog_word_list = set(dialog_word_list)

        dialog_item_embedding = 0.0
        for item in dialog_item_list:
            if item not in item2idx:
                continue
            dialog_item_embedding += item_embedding[item2idx[item]]
        dialog_item_embedding = dialog_item_embedding / len(dialog_item_list) if len(dialog_item_list) > 0 else dialog_item_embedding
        dialog_entity_embedding = 0.0
        for entity in dialog_entity_list:
            if entity not in entity2idx:
                continue
            dialog_entity_embedding += entity_embedding[entity2idx[entity]]
        dialog_entity_embedding = dialog_entity_embedding / len(dialog_entity_list) if len(dialog_entity_list) > 0 else dialog_entity_embedding
        dialog_word_embedding = 0.0
        for word in dialog_word_list:
            if word not in word2idx:
                continue
            dialog_word_embedding += word_embedding[word2idx[word]]
        dialog_word_embedding = dialog_word_embedding / len(dialog_word_list) if len(dialog_word_list) > 0 else dialog_word_embedding
        dialog_embedding[idx] = (dialog_item_embedding + dialog_entity_embedding + dialog_word_embedding) / 3.0
    logger.info("Get dialog embedding")

    return item_embedding, entity_embedding, word_embedding, dialog_embedding

def run_pretrain(task_config:Dict, model_config:Dict) -> None:
    assert task_config["model"] == model_config["model"]
    dataset_name  :str = task_config["dataset"]
    save_path     :str = task_config["save_dir"]

    train_dataset, _, _ = get_dataset(dataset_name)
    logger.info(f"Load dataset {dataset_name}")
    item_edger, entity_edger, word_edger = get_edger(dataset_name)
    logger.info(f"Load item entity and word edger")

    item2idx = json.load(open(os.path.join("data", dataset_name, "entity2id.json"), "r", encoding="utf-8"))
    logger.info(f"Load item2idx   from {os.path.join("data", dataset_name, "entity2id.json")}")
    entity2idx = json.load(open(os.path.join("data", dataset_name, "entity2id.json"), "r", encoding="utf-8"))
    logger.info(f"Load entity2idx from {os.path.join("data", dataset_name, "entity2id.json")}")
    word2idx = json.load(open(os.path.join("data", dataset_name, "token2id.json"), "r", encoding="utf-8"))
    logger.info(f"Load word2idx   from {os.path.join("data", dataset_name, "token2id.json")}")
    item_embedding, entity_embedding, word_embedding, dialog_embedding = train_pretrain(
        task_config, model_config, train_dataset,
        item_edger, entity_edger, word_edger,
        item2idx, entity2idx, word2idx,
    )
    os.makedirs(save_path, exist_ok=True)
    np.save(open(os.path.join(save_path, "item_pretrain.npy"), "wb"), item_embedding)
    logger.info(f"Save item   embedding in {os.path.join(save_path, "item_pretrain.npy")}")
    np.save(open(os.path.join(save_path, "entity_pretrain.npy"), "wb"), entity_embedding)
    logger.info(f"Save entity embedding in {os.path.join(save_path, "entity_pretrain.npy")}")
    np.save(open(os.path.join(save_path, "word_pretrain.npy"), "wb"), word_embedding)
    logger.info(f"Save word   embedding in {os.path.join(save_path, "word_pretrain.npy")}")
    np.save(open(os.path.join(save_path, "dialog_pretrain.npy"), "wb"), dialog_embedding)
    logger.info(f"Save dialog embedding in {os.path.join(save_path, "dialog_pretrain.npy")}")
    return