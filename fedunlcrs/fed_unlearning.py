import os
import collections
import json
import torch
from typing import List, Dict, Tuple, Any
from loguru import logger
import wandb
import pandas as pd
from collections import defaultdict
import copy

from fedunlcrs.model import get_classifer, PretrainEmbeddingModel
from fedunlcrs.evaluate import evaluate_rec

def get_first_GM(task_config: Dict, model_config: Dict) -> None:
    weight_list = []

    dataset_name: str = task_config["dataset"]
    item2idx = json.load(open(os.path.join('data', dataset_name, 'entity2id.json'), "r", encoding="utf-8"))
    entity2idx = json.load(open(os.path.join("data", dataset_name, "entity2id.json"), "r", encoding="utf-8"))
    word2idx = json.load(open(os.path.join("data", dataset_name, "token2id.json"), "r", encoding="utf-8"))
    
    pretrain_model: str = model_config["model"]
    embedding_dim: int = model_config["embedding_dim"]
    
    n_item: int = len(item2idx) + 1  
    n_entity: int = len(entity2idx) + 1
    n_word: int = len(word2idx)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    classifer = get_classifer(pretrain_model)(embedding_dim, n_item)
    model = PretrainEmbeddingModel(n_item, n_entity, n_word, embedding_dim, classifer, device).to(device)
    client_directory = task_config["client_path"]
    
    # 检查目录是否存在
    if not os.path.isdir(client_directory):
        print(f"Error: {client_directory} is not a valid directory.")
        return

    # 遍历目录，加载所有 .pth 文件
    for filename in os.listdir(client_directory):
        if filename.endswith(".pth"):  # 只加载 .pth 文件
            path = os.path.join(client_directory, filename)
            print(f"Loading model from: {path}")
            
            try:
                state_dict = torch.load(path, map_location=device)  # 强制加载到目标设备
                weight_list.append(state_dict)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue  # 跳过无法加载的模型

    weights = [1.0 for _ in weight_list]
    total_weights = sum(weights)

    model.to(device)

    avg_state_dict = {}
    for key in weight_list[0]:  # 遍历第一个模型的所有参数
        avg_weights = sum(weights[i] * weight_list[i][key] for i in range(len(weight_list))) / total_weights
        avg_state_dict[key] = avg_weights

    model.load_state_dict(avg_state_dict)
    output_model_path = "save/unlearning/opendialkg-mlp-8/first_global_model.pth"
    
    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    
    torch.save(model.state_dict(), output_model_path)  # 保存模型权重
    print(f"第一轮全局模型已保存到 {output_model_path}")

def unlearning(task_config: Dict, model_config: Dict) -> None:
    dataset_name: str = task_config["dataset"]
    item2idx = json.load(open(os.path.join('data', dataset_name, 'entity2id.json'), "r", encoding="utf-8"))
    entity2idx = json.load(open(os.path.join("data", dataset_name, "entity2id.json"), "r", encoding="utf-8"))
    word2idx = json.load(open(os.path.join("data", dataset_name, "token2id.json"), "r", encoding="utf-8"))
    
    pretrain_model: str = model_config["model"]
    embedding_dim: int = model_config["embedding_dim"]
    
    n_item: int = len(item2idx) + 1  
    n_entity: int = len(entity2idx) + 1
    n_word: int = len(word2idx)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    classifer = get_classifer(pretrain_model)(embedding_dim, n_item)
    model = PretrainEmbeddingModel(n_item, n_entity, n_word, embedding_dim, classifer, device).to(device)
    
    old_clients = {}  # 用字典存储 {client_id: model_state_dict}
    new_clients = {}  # 用字典存储 {client_id: model_state_dict}
    old_global_models = []
    new_global_models = []
    
    client_path = task_config["client_path"]
    forget_client_id = task_config["forget_client_id"]
    epoch = task_config["calibration_epochs"]

    # 读取初始 old global model
    old_state_dict = torch.load("save/unlearning/opendialkg-mlp-8/first_global_model.pth")
    old_global_models.append(old_state_dict)
    #old state dict是ordereddict
    #old_state_dict的keys:odict_keys(['item_embedding.weight', 'entity_embedding.weight', 'word_embedding.weight', 'classifer.mlp.0.weight', 'classifer.mlp.0.bias', 'classifer.mlp.2.weight', 'classifer.mlp.2.bias'])
    required_keys = old_state_dict.keys()

    # 读取 old clients
    for filename in os.listdir(client_path):
        if filename.endswith(".pth") and f"client_{forget_client_id}" not in filename: 
            path = os.path.join(client_path, filename)
            print(f"Loading model from: {path}")
        
            try:
                state_dict = torch.load(path, map_location=device)
                # 检查是否包含所有必需的键
                missing_keys = [key for key in required_keys if key not in state_dict]
                if missing_keys:
                    print(f"Skipping {filename} due to missing keys: {missing_keys}")
                    continue
                client_id = filename.split(".")[0]
                old_clients[client_id] = state_dict
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue   

    weights = [1.0 for _ in old_clients]  # 每个 client 的权重
    total_weights = sum(weights)

    model.to(device)

    # 计算初始 new global model
    avg_state_dict = collections.OrderedDict()
    for key in required_keys:  # 使用全局模型的键集合
        # 假设所有old_clients的state_dict都包含key（已通过检查）
        avg_weights = sum(weights[i] * old_clients[client_id][key] for i, client_id in enumerate(old_clients)) / total_weights
        avg_state_dict[key] = avg_weights

    model.load_state_dict(avg_state_dict)  # 载入到 model
    new_global_models.append(avg_state_dict)
    # 生成初始 new_clients
    for client_id in old_clients:
        new_clients[client_id] = copy.deepcopy(avg_state_dict)  # 所有 new_clients 共享新 global model
        #此时的new clients是dict，new_clients[client_id]是ordereddict

    #print(new_clients)

    # 记录 old_clients 和 new_clients
    #old clients和new clients都是dict， old clients list是存着不同dict的list
    old_clients_list = []
    new_clients_list = []
    old_clients_list.append(old_clients)
    new_clients_list.append(new_clients)

    #print(old_clients_list)

    # 训练 epoch 轮
    for ii in range(epoch):
            return_model_state = unlearning_one_step(
                old_clients_list[ii], new_clients_list[ii], old_global_models[ii], new_global_models[ii]
        )
            temp = return_model_state
            old_clients_list.append(new_clients_list[ii])
            for client_id in new_clients_list[ii]:
                new_clients_list[ii][client_id] = copy.deepcopy(temp)
            new_clients_list.append(new_clients_list[ii])
            old_global_models.append(new_global_models[ii])
            new_global_models.append(return_model_state)

    print(new_global_models[-1])
    #print("test1")


def unlearning_one_step(old_clients: Dict, 
                        new_clients: Dict, 
                        old_global_model: collections.OrderedDict, 
                        new_global_model: collections.OrderedDict):
    old_param_update = collections.OrderedDict(
        {layer: torch.zeros_like(param) for layer, param in old_global_model.items()}
    )
    new_param_update = collections.OrderedDict(
        {layer: torch.zeros_like(param) for layer, param in old_global_model.items()}
    )
    return_model_state = collections.OrderedDict(
        {layer: torch.zeros_like(param) for layer, param in old_global_model.items()}
    )
    new_global_model_state = new_global_model

    assert len(old_clients) == len(new_clients)
    for client_id in new_clients:  # 遍历 client_id
        for layer in old_global_model.keys():
            old_param_update[layer] += old_clients[client_id][layer]
            new_param_update[layer] += new_clients[client_id][layer]

    for layer in old_global_model.keys():
        old_param_update[layer] /= len(new_clients)
        new_param_update[layer] /= len(new_clients)

        old_param_update[layer] = old_param_update[layer] - old_global_model[layer]
        new_param_update[layer] = new_param_update[layer] - new_global_model[layer]

        length = torch.norm(old_param_update[layer])
        direction = new_param_update[layer] / torch.norm(new_param_update[layer])

        return_model_state[layer] = new_global_model_state[layer] + length * direction
    
    #print(type(return_model_state))
    
    return return_model_state


def run_unlearning(task_config: Dict, model_config: Dict) -> None:
    get_first_GM(task_config, model_config)
    unlearning(task_config, model_config)
