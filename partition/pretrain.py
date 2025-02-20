import torch
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
    print(f"[+] Get dataloader with size of {len(dataloader)}.")

    n_item = len(item2idx) + 1
    n_entity = len(entity2idx) + 1
    n_word = len(word2idx)
    embedding_dim = 128
    device = "cuda:0"
    epochs = 0

    classifer = get_classifer(classifer_model)(embedding_dim, n_item)
    model = PretrainEmbeddingModel(n_item, n_entity, n_word, embedding_dim, classifer, device).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
        print(epoch, tot_loss / len(dataloader))
    return model.item_embedding.weight.detach().cpu().numpy(), model.entity_embedding.weight.detach().cpu().numpy(), model.word_embedding.weight.detach().cpu().numpy()