import os
import json
from tqdm import tqdm
from loguru import logger
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fedunlcrs.utils import get_dataloader, get_dataset, get_edger
from fedunlcrs.model import get_classifer

class PretrainEmbeddingModel(torch.nn.Module):
    def __init__(
            self, n_item:int, n_entity:int, n_word:int,
            embedding_dim:int, classifer:torch.nn.Module, device:str
        ) -> None:
        super().__init__()
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_word = n_word
        self.embedding_dim = embedding_dim
        self.item_embedding = torch.nn.Embedding(n_item, embedding_dim)
        self.entity_embedding = torch.nn.Embedding(n_entity, embedding_dim)
        self.word_embedding = torch.nn.Embedding(n_word, embedding_dim)
        self.classifer = classifer
        self.device = device
        return
    
    def forward(
            self, item_list:torch.LongTensor, entity_list:torch.LongTensor, word_list:torch.LongTensor,
            item_edger:Dict, entity_edger:Dict, word_edger:Dict
        ) -> torch.FloatTensor:
        item_emb = self.item_embedding(item_list).mean(0, keepdim=True) if len(item_list) > 0 else torch.zeros((1, self.embedding_dim)).to(self.device)
        entity_emb = self.entity_embedding(entity_list).mean(0, keepdim=True) if len(entity_list) > 0 else torch.zeros((1, self.embedding_dim)).to(self.device)
        word_emb = self.word_embedding(word_list).mean(0, keepdim=True) if len(word_list) > 0 else torch.zeros((1, self.embedding_dim)).to(self.device)
        emb = (item_emb + entity_emb + word_emb) / 3.0
        out = self.classifer(emb, item_edger, entity_edger, word_edger)
        return out

def get_sub_dataloader(
        tot_dataloader:List, sub_dataset_mask:Dict,
        split_item:bool, split_entity:bool, split_word:bool, split_dialog:bool
    ) -> List:
    sub_dataloader = []
    item_mask   = sub_dataset_mask["item_mask"]
    entity_mask = sub_dataset_mask["entity_mask"]
    word_mask   = sub_dataset_mask["word_mask"]
    dialog_mask = sub_dataset_mask["dialog_mask"]

    for dialog_idx, raw_meta_data in enumerate(tot_dataloader):
        if dialog_idx not in dialog_mask and split_dialog:
            continue
        res_meta_data = {}

        res_item_list = []
        for item in raw_meta_data["item"]:
            if item in item_mask or not split_item:
                res_item_list.append(item)
        res_meta_data["item"] = res_item_list

        res_entity_list = []
        for entity in raw_meta_data["entity"]:
            if entity in entity_mask or not split_entity:
                res_entity_list.append(entity)
        res_meta_data["entity"] = res_entity_list

        res_word_list = []
        for word in raw_meta_data["word"]:
            if word in word_mask or not split_word:
                res_word_list.append(word)
        res_meta_data["word"] = res_word_list

        res_meta_data["label"] = raw_meta_data["label"]
        sub_dataloader.append(res_meta_data)

    return sub_dataloader

def train_federated(
        model:torch.nn.Module, dataloader:List, optimizer:torch.optim.Optimizer, criterion:torch.nn.Module,
        item_edger:Dict, entity_edger:Dict, word_edger:Dict, rank:int) -> Tuple[torch.nn.Module, float]:
    tot_loss = 0.0
    device = model.device
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
        tot_loss += loss
    return model, tot_loss/len(dataloader)

def aggregate_models(model:torch.nn.Module) -> None:
    params = [p for p in model.parameters()]
    for param in params:
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= dist.get_world_size()
    return

def work_federated(rank:int, task_config:Dict, model_config:Dict) -> None:
    assert task_config["model"] == model_config["model"]
    dataset_name  :str = task_config["dataset"]
    mask_dir      :str = task_config["mask_dir"]
    save_path     :str = task_config["save_dir"]
    n_client    :int   = task_config["n_client"]

    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:12345",
        world_size=n_client,
        rank=rank,
    )

    train_dataset, valid_dataset, test_dataset = get_dataset(dataset_name)
    if rank == 0:
        logger.info(f"Load dataset {dataset_name}")
    item_edger, entity_edger, word_edger = get_edger(dataset_name)
    if rank == 0:
        logger.info(f"Load item entity and word edger")

    item2idx = json.load(open(os.path.join("data", dataset_name, "entity2id.json"), "r", encoding="utf-8"))
    if rank == 0:
        logger.info(f"Load item2idx   from {os.path.join("data", dataset_name, "entity2id.json")}")
    entity2idx = json.load(open(os.path.join("data", dataset_name, "entity2id.json"), "r", encoding="utf-8"))
    if rank == 0:
        logger.info(f"Load entity2idx from {os.path.join("data", dataset_name, "entity2id.json")}")
    word2idx = json.load(open(os.path.join("data", dataset_name, "token2id.json"), "r", encoding="utf-8"))
    if rank == 0:
        logger.info(f"Load word2idx   from {os.path.join("data", dataset_name, "token2id.json")}")

    tot_train_dataloader = get_dataloader(train_dataset, item2idx, entity2idx, word2idx)
    tot_valid_dataloader = get_dataloader(valid_dataset, item2idx, entity2idx, word2idx)
    tot_test_dataloader  = get_dataloader(test_dataset, item2idx, entity2idx, word2idx)
    if rank == 0:
        logger.info(f"Build total dataloader")

    pretrain_model:str   = model_config["model"]
    embedding_dim :int   = model_config["embedding_dim"]
    n_item        :int   = len(item2idx) + 1
    n_entity      :int   = len(entity2idx) + 1
    n_word        :int   = len(word2idx)
    epochs        :int   = task_config["epochs"]
    learning_rate :float = float(task_config["learning_rate"])
    split_item    :bool  = task_config["split_item"]
    split_entity  :bool  = task_config["split_entity"]
    split_word    :bool  = task_config["split_word"]
    split_dialog  :bool  = task_config["split_dialog"]
    device        :str   = f"cuda:{rank}"

    criterion = torch.nn.CrossEntropyLoss()
    classifer = get_classifer(pretrain_model)(embedding_dim, n_item)
    model = PretrainEmbeddingModel(n_item, n_entity, n_word, embedding_dim, classifer, device).to(device)
    if rank == 0:
        logger.info(f"Build model:\n{model}")
    sub_dataset_mask = json.load(open(os.path.join(mask_dir, f"sub_dataset_mask_{rank+1}_{n_client}.json"), encoding="utf-8"))
    logger.info(f"Client {rank + 1} load sub dataset mask from {os.path.join(mask_dir, f"sub_dataset_mask_{rank+1}_{n_client}.json")}")
    sub_train_dataloader = get_sub_dataloader(
        tot_train_dataloader, sub_dataset_mask,
        split_item, split_entity, split_word, split_dialog,
    )
    sub_valid_dataloader = get_sub_dataloader(
        tot_valid_dataloader, sub_dataset_mask,
        split_item, split_entity, split_word, split_dialog,
    )
    sub_test_dataloader  = get_sub_dataloader(
        tot_test_dataloader,  sub_dataset_mask,
        split_item, split_entity, split_word, split_dialog,
    )
    if rank == 0:
        logger.info(f"Build sub dataloader")

    if rank == 0:
        logger.info("Start federated training")
    for epoch in range(1, epochs + 1, 1):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        client_model, client_loss = train_federated(
            model, sub_train_dataloader, optimizer, criterion,
            item_edger, entity_edger, word_edger, rank
        )
        aggregate_models(model)
        logger.info(f"[Epoch:{epoch}/{epochs}] Client {rank + 1} average loss:{client_loss:.6f}]")
        if rank == 0:
            logger.info(f"Aggregate models")

    if rank == 0:
        logger.info("Finish federated training")
    dist.destroy_process_group()
    return

def run_federated(task_config:Dict, model_config:Dict) -> None:
    world_size = task_config["n_client"]
    mp.spawn(work_federated, nprocs=world_size, args=(task_config, model_config))
    return