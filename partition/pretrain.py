import torch
import numpy as np
from tqdm import tqdm

from utils import get_dataloader
from partition.classifer import get_classifer

class PretrainEmbeddingModel(torch.nn.Module):
    def __init__(self, n_item, n_entity, n_word, embedding_dim, classifer, device) -> None:
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
    
    def forward(self, item_list, entity_list, word_list):
        item_emb = self.item_embedding(item_list).mean(0, keepdim=True) if len(item_list) > 0 else torch.zeros((1, self.embedding_dim)).to(self.device)
        entity_emb = self.entity_embedding(entity_list).mean(0, keepdim=True) if len(entity_list) > 0 else torch.zeros((1, self.embedding_dim)).to(self.device)
        word_emb = self.word_embedding(word_list).mean(0, keepdim=True) if len(word_list) > 0 else torch.zeros((1, self.embedding_dim)).to(self.device)
        emb = (item_emb + entity_emb + word_emb) / 3.0
        out = self.classifer(emb)
        return out

def run_pretrain(dataset, classifer_model, train_dataset, item_edger, entity_edger, word_edger, item2idx, entity2idx, word2idx):
    dataloader = get_dataloader(train_dataset, item2idx, entity2idx, word2idx)
    print(f"[+] Build dataloader with size of {len(dataloader)}")

    n_item = len(item2idx) + 1
    n_entity = len(entity2idx) + 1
    n_word = len(word2idx)
    embedding_dim = 128
    device = "cuda:0"
    epochs = 4

    classifer = get_classifer(classifer_model)(embedding_dim, n_item)
    model = PretrainEmbeddingModel(n_item, n_entity, n_word, embedding_dim, classifer, device).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"[+] Build PretrainModel")
    print(f"[+] {model}")

    print(f"[T] Start pretrain training")
    for epoch in range(1, epochs + 1, 1):
        tot_loss = 0.0
        for meta_data in tqdm(dataloader):
            item_list = torch.LongTensor(meta_data["item"]).to(device)
            entity_list = torch.LongTensor(meta_data["entity"]).to(device)
            word_list = torch.LongTensor(meta_data["word"]).to(device)
            label = torch.LongTensor(meta_data["label"]).to(device)

            output = model(item_list, entity_list, word_list)
            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            tot_loss += loss.cpu().item()
        print(f"[T] Epoch:{epoch}/{epochs} Loss:{tot_loss / len(dataloader)}")
    print(f"[T] End pretrain training")

    item_embedding = model.item_embedding.weight.detach().cpu().numpy()
    entity_embedding = model.entity_embedding.weight.detach().cpu().numpy()
    word_embedding = model.word_embedding.weight.detach().cpu().numpy()

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

    return item_embedding, entity_embedding, word_embedding, dialog_embedding