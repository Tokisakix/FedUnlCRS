import os
import json
import torch
from tqdm import tqdm
from loguru import logger
from typing import Any, Dict, List, Tuple

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

    for dialog_idx, raw_meta_data in enumerate(tqdm(tot_dataloader)):
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
        item_edger:Dict, entity_edger:Dict, word_edger:Dict, federated_tqdm:Any) -> Tuple[torch.nn.Module, float]:
    tot_loss = 0.0
    step = 0
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
        step += 1
        federated_tqdm.set_postfix(loss=f"{tot_loss/step:.6f}")
        federated_tqdm.update(1)
        #DEBUG!
        # break
    return model, tot_loss/len(dataloader)

def aggregate_models(client_models: List[torch.nn.Module]) -> List[torch.nn.Module]:
    weights = [model.state_dict() for model in client_models]
    avg_weights = {}

    for key in weights[0]:
        layer_weights = [w[key] for w in weights]
        avg_weights[key] = torch.mean(torch.stack(layer_weights), dim=0)
    for model in client_models:
        model.load_state_dict(avg_weights)

    return client_models


def run_federated(task_config:Dict, model_config:Dict) -> None:
    assert task_config["model"] == model_config["model"]
    dataset_name  :str = task_config["dataset"]
    mask_dir      :str = task_config["mask_dir"]
    save_path     :str = task_config["save_dir"]

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
    device        :str   = task_config["device"]
    epochs        :int   = task_config["epochs"]
    n_clusters    :int   = task_config["n_clusters"]
    learning_rate :float = float(task_config["learning_rate"])
    split_item    :bool  = task_config["split_item"]
    split_entity  :bool  = task_config["split_entity"]
    split_word    :bool  = task_config["split_word"]
    split_dialog  :bool  = task_config["split_dialog"]

    models = []
    sub_dataloaders = []
    criterion = torch.nn.CrossEntropyLoss()

    for idx in range(1, n_clusters + 1, 1):
        classifer = get_classifer(pretrain_model)(embedding_dim, n_item)
        model = PretrainEmbeddingModel(n_item, n_entity, n_word, embedding_dim, classifer, device).to(device)
        logger.info(f"Client {idx} build model:\n{model}")
        sub_dataset_mask = json.load(open(os.path.join(mask_dir, f"sub_dataset_mask_{idx}_{n_clusters}.json"), encoding="utf-8"))
        logger.info(f"Client {idx} load sub dataset mask from {os.path.join(mask_dir, f"sub_dataset_mask_{idx}_{n_clusters}.json")}")
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
        logger.info(f"Client {idx} build sub dataloader")
        models.append(model)
        sub_dataloaders.append((sub_train_dataloader, sub_valid_dataloader, sub_test_dataloader))

    logger.info("Start federated training")
    for epoch in range(1, epochs + 1, 1):
        tot_client_loss = 0.0

        tot_client_models = []
        for idx in range(n_clusters):
            optimizer = torch.optim.Adam(models[idx].parameters(), lr=learning_rate)
            sub_train_dataloader = sub_dataloaders[idx][0]
            with tqdm(total= len(sub_train_dataloader)) as federated_tqdm:
                federated_tqdm.set_description(f"Client {idx + 1} Epoch: {epoch}/{epochs}")
                client_model, client_loss = train_federated(
                    models[idx], sub_train_dataloader, optimizer, criterion,
                    item_edger, entity_edger, word_edger, federated_tqdm
                )
                tot_client_models.append(client_model)
                tot_client_loss += client_loss
        logger.info(f"[Epoch:{epoch}/{epochs}] Client average loss:{tot_client_loss/n_clusters:.6f}]")

        models = aggregate_models(tot_client_models)
        logger.info(f"Aggregate models")

    logger.info("Finish pretraining")

    return