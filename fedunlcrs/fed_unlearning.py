import os
import copy
import json
import torch
import torch.distributed as dist
from typing import List, Dict, Tuple, Any
from loguru import logger
import wandb
import pandas as pd
from collections import defaultdict

from fedunlcrs.utils import get_dataloader, get_dataset, get_edger
from fedunlcrs.model import get_classifer, PretrainEmbeddingModel
from fedunlcrs.evaluate import evaluate_rec
from fedunlcrs.federated import aggregate_models, run_federated, get_sub_dataloader, train_federated

def work_federated_with_history(rank: int, task_config: Dict, model_config: Dict) -> Tuple[List[Dict], List[Dict]]:
    """增强版联邦训练，保存每轮全局模型和客户端模型参数"""
    # 初始化部分与原代码完全兼容
    global_hist = []  # 每轮的全局模型参数（聚合后）
    client_hist = []  # 每轮的客户端本地模型参数（聚合前）

    # 原代码初始化部分（保持完全一致）
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
        # 训练阶段（与原代码完全一致）
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        client_model, client_loss = train_federated(
            model, sub_train_dataloader, optimizer, criterion,
            item_edger, entity_edger, word_edger
        )

        # 记录客户端本地模型（聚合前）
        client_hist.append(copy.deepcopy(model.state_dict()))  # 新增历史记录

        # 损失收集与日志（保持原逻辑）
        loss_list = [torch.zeros_like(client_loss) for _ in range(n_client)]
        dist.all_gather(loss_list, client_loss)
        dist.all_reduce(client_loss, dist.ReduceOp.SUM)
        avg_loss = client_loss / dist.get_world_size()
        logger.info(f"[Epoch:{epoch}/{epochs}] Avg client loss: {avg_loss.item():.6f}")

        loss_values = [tensor.item() for tensor in loss_list]
        loss_df = pd.DataFrame([[f"{loss:.6f}" for loss in loss_values]], columns=[f"Client {i+1}" for i in range(n_client)])
        loss_df.insert(0, "Client", ["Loss"])
        logger.info(f"\n{loss_df.to_string(index=False)}")

        # 模型聚合（保持原逻辑）
        aggregate_models(model)
        logger.info(f"Aggregate models")

        # 记录全局模型（聚合后）
        global_hist.append(copy.deepcopy(model.state_dict()))  # 新增历史记录

        # 验证与日志（保持原逻辑）
        if rank == 0:
            logger.info("Evaluate model in mode [Valid]")
            evaluate_res = evaluate_rec(
                model, sub_valid_dataloader,
                item_edger, entity_edger, word_edger
            )
            for meta_res in evaluate_res:
                meta_df = pd.DataFrame([meta_res])
                logger.info(f"\n{meta_df.to_string(index=False)}")

        # Wandb日志（保持原逻辑）
        if rank == 0 and task_config["wandb_use"]:
            wandb_log = {
                "epoch":epoch,
            }
            for idx, loss in enumerate(loss_values):
                wandb_log[f"client_{idx+1}_loss"] = loss
            wandb.log(wandb_log)
    
    logger.info("Finish federated training")

    # 测试评估（保持原逻辑）
    if rank == 0:
        logger.info("Evaluate model in mode [Test]")
        evaluate_res = evaluate_rec(
            model, sub_test_dataloader,
            item_edger, entity_edger, word_edger
        )
        for meta_res in evaluate_res:
            meta_df = pd.DataFrame([meta_res])
            logger.info(f"\n{meta_df.to_string(index=False)}")

    # 模型保存（新增历史参数保存）
    torch.save({
        "global_hist": global_hist,
        "client_hist": client_hist,
        "final_model": model.state_dict()
    }, os.path.join(save_path, f"client_{rank+1}_full_hist.pth"))
    logger.info(f"Save full history in {os.path.join(save_path, 'client_<client_id>_full_hist.pth')}")

    if rank == 0 and task_config["wandb_use"]:
        wandb.finish()

    dist.destroy_process_group()
    
    return global_hist, client_hist


def federated_unlearning(
    global_hist: List[Dict],
    client_hist: List[Dict], 
    forget_id: int,
    task_config: Dict,
    model_config: Dict
) -> torch.nn.Module:
    """联邦去学习完整实现（与原始联邦学习对称设计）
    
    Args:
        global_hist: 各轮次全局模型参数列表
            [
                {"layer1.weight": tensor1, "layer1.bias": tensor2, ...},  # 第1轮参数
                {"layer1.weight": tensor1, "layer1.bias": tensor2, ...},  # 第2轮参数
                ...
            ]
        client_hist: 各轮次客户端本地模型参数列表（结构同global_hist）
        forget_id: 需要遗忘的客户端ID（1-based）
        task_config: 任务配置字典，需包含：
            - n_client: 总客户端数
            - calibration_epochs: 校准训练轮数（可选）
            - calibration_lr: 校准学习率（可选）
        model_config: 模型配置字典
    
    Returns:
        去学习后的全局模型实例
    """
    # ----------------------
    # 阶段0：参数校验与初始化
    # ----------------------
    # 参数有效性检查
    n_client = task_config["n_client"]
    if forget_id < 1 or forget_id > n_client:
        raise ValueError(f"无效客户端ID {forget_id}，有效范围1-{n_client}")
    forget_idx = forget_id - 1  # 转换为0-based索引

    # 设备设置（与原联邦学习对称）
    device = f"cuda:{forget_idx}" if torch.cuda.is_available() else "cpu"
    
    # ----------------------
    # 阶段1：逆向参数修正
    # ----------------------
    # 初始化模型（与原代码完全一致）
    model = PretrainEmbeddingModel(
        n_item=model_config["n_item"],
        n_entity=model_config["n_entity"],
        n_word=model_config["n_word"],
        embedding_dim=model_config["embedding_dim"],
        classifer=get_classifer(model_config["model"])(model_config["embedding_dim"], model_config["n_item"]),
        device=device
    ).to(device)

    # 逆向计算各轮次全局参数
    corrected_models = []
    for global_state, client_state in zip(global_hist, client_hist):
        new_global = {}
        for key in global_state.keys():
            # 逆向计算公式：(原全局参数 * 总客户端数 - 被遗忘客户端参数) / 剩余客户端数
            numerator = global_state[key] * n_client - client_state[key]
            new_global[key] = numerator / (n_client - 1)
        corrected_models.append(new_global)

    # 加载最终修正模型
    model.load_state_dict(corrected_models[-1])

    # ----------------------
    # 阶段2：校准训练
    # ----------------------
    if task_config.get("calibration_epochs", 0) > 0:
        # 初始化训练组件（与原训练逻辑对称）
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=task_config.get("calibration_lr", 1e-4)
        )
        criterion = torch.nn.CrossEntropyLoss()

        # 获取保留客户端数据（与原数据加载逻辑对称）
        retained_loader = get_retained_dataloader(forget_id, task_config)

        # 校准训练循环（与train_federated结构对称）
        for epoch in range(task_config["calibration_epochs"]):
            model.train()
            total_loss = 0.0
            
            for batch in retained_loader:
                # 数据预处理（与原训练完全一致）
                items = torch.LongTensor(batch["item"]).to(device)
                entities = torch.LongTensor(batch["entity"]).to(device)
                words = torch.LongTensor(batch["word"]).to(device)
                labels = torch.LongTensor(batch["label"]).to(device)

                # 前向传播
                outputs = model(items, entities, words, 
                               task_config["item_edger"],
                               task_config["entity_edger"],
                               task_config["word_edger"])

                # 损失计算与反向传播
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # 打印校准训练日志（新增）
            avg_loss = total_loss / len(retained_loader)
            print(f"校准训练轮次 [{epoch+1}/{task_config['calibration_epochs']}] 平均损失: {avg_loss:.4f}")

    # ----------------------
    # 阶段3：最终模型验证
    # ----------------------
    # 与原评估逻辑对称
    if task_config.get("validate_after_unlearn", True):
        test_loader = get_retained_dataloader(forget_id, task_config, mode="test")
        evaluate_rec(model, test_loader, 
                   task_config["item_edger"],
                   task_config["entity_edger"],
                   task_config["word_edger"])

    return model


def get_retained_dataloader(forget_id: int, task_config: Dict[str, Any]) -> List[Dict[str, List[int]]]:
    """构建保留客户端的联合数据加载器
    
    Args:
        forget_id (int): 要遗忘的客户端ID (1-based)
        task_config (Dict): 任务配置字典，需包含：
            - mask_dir: mask文件目录
            - n_client: 总客户端数
            - split_item: 是否分割item
            - split_entity: 是否分割entity
            - split_word: 是否分割word
            - split_dialog: 是否分割dialog
    
    Returns:
        List[Dict]: 合并后的数据加载器，格式与原始数据加载器一致
    
    Raises:
        FileNotFoundError: 当mask文件不存在时抛出
        ValueError: 当forget_id无效时抛出
    """
    # 参数校验
    n_client = task_config["n_client"]
    if forget_id < 1 or forget_id > n_client:
        raise ValueError(f"Invalid forget_id {forget_id}, must be 1~{n_client}")

    # 初始化合并后的mask
    merged_mask = defaultdict(set)
    mask_keys = ["item_mask", "entity_mask", "word_mask", "dialog_mask"]

    # 遍历所有客户端
    for client_id in range(1, n_client+1):
        if client_id == forget_id:
            continue

        # 加载mask文件
        mask_path = os.path.join(
            task_config["mask_dir"],
            f"sub_dataset_mask_{client_id}_{n_client}.json"
        )
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        with open(mask_path, "r", encoding="utf-8") as f:
            client_mask = json.load(f)

        # 合并各mask字段
        for key in mask_keys:
            if key in client_mask:
                merged_mask[key].update(set(client_mask[key]))

    # 转换set为list
    final_mask = {
        key: list(merged_mask[key]) 
        for key in mask_keys
    }

    # 获取原始数据集
    dataset_name = task_config["dataset"]
    train_dataset, _, _ = get_dataset(dataset_name)
    
    # 加载索引映射
    item2idx = json.load(open(os.path.join("data", dataset_name, "entity2id.json"), "r", encoding="utf-8"))
    entity2idx = json.load(open(os.path.join("data", dataset_name, "entity2id.json"), "r", encoding="utf-8"))
    word2idx = json.load(open(os.path.join("data", dataset_name, "token2id.json"), "r", encoding="utf-8"))

    # 生成总数据加载器
    tot_dataloader = get_dataloader(train_dataset, item2idx, entity2idx, word2idx)

    # 生成保留客户端联合数据加载器
    return get_sub_dataloader(
        tot_dataloader=tot_dataloader,
        sub_dataset_mask=final_mask,
        split_item=task_config["split_item"],
        split_entity=task_config["split_entity"],
        split_word=task_config["split_word"],
        split_dialog=task_config["split_dialog"]
    )

def run_unlearning_pipeline():
    """完整的联邦去学习工作流示例（与原联邦学习流程对称）
    
    步骤：
    1. 运行带历史记录的联邦学习训练
    2. 执行去学习操作移除指定客户端影响
    3. 评估去学习后模型性能
    
    返回：
        torch.nn.Module: 去学习后的全局模型
    """
    # ----------------------
    # 阶段1：初始化配置
    # ----------------------
    # 任务配置（与原联邦学习配置兼容）
    task_config = {
        "dataset": "amazon",          # 数据集名称
        "n_client": 5,                # 总客户端数
        "epochs": 10,                 # 联邦训练轮次
        "calibration_epochs": 3,      # 校准训练轮次
        "calibration_lr": 1e-4,       # 校准学习率
        "mask_dir": "./masks",        # 客户端掩码文件目录
        "save_dir": "./saved_models", # 模型保存路径
        "split_item": True,           # 是否分割商品特征
        "split_entity": False,        # 是否分割实体特征
        "split_word": True,           # 是否分割文本特征
        "split_dialog": True,         # 是否分割对话特征
        "wandb_use": True,            # 是否使用Wandb日志
        "wandb_project": "FedUnlearn",# Wandb项目名称
        "wandb_name": "unlearn_exp1"  # Wandb实验名称
    }
    
    # 模型配置（与原模型配置一致）
    model_config = {
        "model": "TwoTower",          # 模型类型
        "embedding_dim": 128,         # 嵌入维度
        "n_item": 10000,              # 商品总数（根据数据集自动获取）
        "n_entity": 5000,             # 实体总数
        "n_word": 30000               # 词汇表大小
    }
    
    # ----------------------
    # 阶段2：联邦训练（带历史记录）
    # ----------------------
    print("\n===== 开始联邦训练 =====")
    global_hist, client_hist = run_federated(task_config, model_config)
    
    # ----------------------
    # 阶段3：联邦去学习
    # ----------------------
    print("\n===== 开始联邦去学习 =====")
    # 指定要遗忘的客户端ID（1-based）
    forget_client_id = 2
    
    # 执行去学习
    unlearned_model = federated_unlearning(
        global_hist=global_hist,
        client_hist=client_hist,
        forget_id=forget_client_id,
        task_config=task_config,
        model_config=model_config
    )
    
    # ----------------------
    # 阶段4：模型评估
    # ----------------------
    print("\n===== 评估去学习模型 =====")
    # 加载测试数据集
    _, _, test_dataset = get_dataset(task_config["dataset"])
    item2idx = json.load(open(f"data/{task_config['dataset']}/entity2id.json"))
    
    # 生成保留客户端测试集（排除被遗忘客户端）
    test_loader = get_retained_dataloader(
        forget_id=forget_client_id,
        task_config=task_config,
        mode="test"  # 新增模式参数区分训练/测试
    )
    
    # 执行评估
    evaluation_results = evaluate_rec(
        model=unlearned_model,
        dataloader=test_loader,
        item_edger=get_edger(task_config["dataset"])[0],
        entity_edger=get_edger(task_config["dataset"])[1],
        word_edger=get_edger(task_config["dataset"])[2]
    )
    
    # 打印评估结果
    print("\n评估结果:")
    for metric in evaluation_results:
        print(f"{metric['name']}:")
        print(f"  Recall@10: {metric['recall@10']:.4f}")
        print(f"  NDCG@10: {metric['ndcg@10']:.4f}")
    
    return unlearned_model