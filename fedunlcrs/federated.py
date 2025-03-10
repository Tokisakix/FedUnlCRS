import os
import json
import wandb
import pandas as pd
from tqdm import tqdm
from loguru import logger
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fedunlcrs.utils import get_dataloader, get_dataset, get_edger
from fedunlcrs.model import get_classifer, PretrainEmbeddingModel
from fedunlcrs.evaluate import evaluate_rec

def get_sub_dataloader(
        tot_dataloader:List, sub_dataset_mask:Dict,
        split_item:bool, split_entity:bool, split_word:bool, split_dialog:bool
    ) -> List:
    sub_dataloader = []
    item_mask   = sub_dataset_mask["item_mask"]
    entity_mask = sub_dataset_mask["entity_mask"]
    word_mask   = sub_dataset_mask["word_mask"]
    dialog_mask = sub_dataset_mask["dialog_mask"]

    rank = dist.get_rank()
    for dialog_idx, raw_meta_data in enumerate(tqdm(tot_dataloader, disable=(rank != 0))):
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
        item_edger:Dict, entity_edger:Dict, word_edger:Dict) -> Tuple[torch.nn.Module, torch.FloatTensor]:
    tot_loss = 0.0
    device = model.device
    rank = dist.get_rank()
    for meta_data in tqdm(dataloader, disable=(rank != 0)):
        item_list = torch.LongTensor(meta_data["item"]).to(device)
        entity_list = torch.LongTensor(meta_data["entity"]).to(device)
        word_list = torch.LongTensor(meta_data["word"]).to(device)
        label = torch.LongTensor(meta_data["label"]).to(device)

        output = model(item_list, entity_list, word_list, item_edger, entity_edger, word_edger)
        optimizer.zero_grad()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        loss = loss.detach()
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
    n_client      :int   = task_config["n_client"]

    if rank != 0:
        logger.remove()
    elif task_config["wandb_use"]:
        wandb.init(project=task_config["wandb_project"], name=task_config["wandb_name"])
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:12345",
        world_size=n_client,
        rank=rank,
    )

    train_dataset, valid_dataset, test_dataset = get_dataset(dataset_name)
    logger.info(f"Load dataset {dataset_name}")
    item_edger, entity_edger, word_edger = get_edger(dataset_name)
    logger.info(f"Load item entity and word edger")

    item2idx = json.load(open(os.path.join("data", dataset_name, "entity2id.json"), "r", encoding="utf-8"))
    logger.info(f"Load item2idx   from {os.path.join("data", dataset_name, "entity2id.json")}")
    entity2idx = json.load(open(os.path.join("data", dataset_name, "entity2id.json"), "r", encoding="utf-8"))
    logger.info(f"Load entity2idx from {os.path.join("data", dataset_name, "entity2id.json")}")
    word2idx = json.load(open(os.path.join("data", dataset_name, "token2id.json"), "r", encoding="utf-8"))
    logger.info(f"Load word2idx   from {os.path.join("data", dataset_name, "token2id.json")}")

    tot_train_dataloader = get_dataloader(train_dataset, item2idx, entity2idx, word2idx)
    tot_valid_dataloader = get_dataloader(valid_dataset, item2idx, entity2idx, word2idx)
    tot_test_dataloader  = get_dataloader(test_dataset, item2idx, entity2idx, word2idx)
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
    logger.info(f"Build model:\n{model}")
    sub_dataset_mask = json.load(open(os.path.join(mask_dir, f"sub_dataset_mask_{rank+1}_{n_client}.json"), encoding="utf-8"))
    logger.info(f"Load sub dataset mask from {os.path.join(mask_dir, f"sub_dataset_mask_<client_id>_{n_client}.json")}")
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
    logger.info(f"Build sub dataloader")

    logger.info("Start federated training")
    for epoch in range(1, epochs + 1, 1):
        # train worker
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        client_model, client_loss = train_federated(
            model, sub_train_dataloader, optimizer, criterion,
            item_edger, entity_edger, word_edger
        )

        # Loss display
        loss_list = [torch.zeros_like(client_loss) for _ in range(n_client)]
        dist.all_gather(loss_list, client_loss)
        dist.all_reduce(client_loss, dist.ReduceOp.SUM)
        avg_loss = client_loss / dist.get_world_size()
        logger.info(f"[Epoch:{epoch}/{epochs}] Avg client loss: {avg_loss.item():.6f}")

        loss_values = [tensor.item() for tensor in loss_list]
        loss_df = pd.DataFrame([[f"{loss:.6f}" for loss in loss_values]], columns=[f"Client {i+1}" for i in range(n_client)])
        loss_df.insert(0, "Client", ["Loss"])
        logger.info(f"\n{loss_df.to_string(index=False)}")

        # Aggregate models
        aggregate_models(model)
        logger.info(f"Aggregate models")

        # Evaluate valid data
        if rank == 0:
            logger.info("Evaluate model in mode [Valid]")
            evaluate_res = evaluate_rec(
                model, sub_valid_dataloader,
                item_edger, entity_edger, word_edger
            )
            evaluate_df = pd.DataFrame([evaluate_res])
            logger.info(f"\n{evaluate_df.to_string(index=False)}")

        # Wandb
        if rank == 0 and task_config["wandb_use"]:
            wandb_log = {
                "epoch":epoch,
            }
            for idx, loss in enumerate(loss_values):
                wandb_log[f"client_{idx+1}_loss"] = loss
            wandb.log(wandb_log)
    logger.info("Finish federated training")

    # Evaluate test data
    if rank == 0:
        logger.info("Evaluate model in mode [Test]")
        evaluate_res = evaluate_rec(
            model, sub_test_dataloader,
            item_edger, entity_edger, word_edger
        )
        evaluate_df = pd.DataFrame([evaluate_res])
        logger.info(f"\n{evaluate_df.to_string(index=False)}")

    # Save model
    torch.save(model.state_dict(), os.path.join(save_path, f"client_{rank+1}_model.pth"))
    logger.info(f"Save client model in {os.path.join(save_path, "client_<client_id>_model.pth")}")
    if rank == 0 and task_config["wandb_use"]:
        wandb.finish()

    dist.destroy_process_group()
    return

def run_federated(task_config:Dict, model_config:Dict) -> None:
    world_size = task_config["n_client"]
    mp.spawn(work_federated, nprocs=world_size, args=(task_config, model_config), daemon=True)
    return