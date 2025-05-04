import torch
import torch.nn.functional as F
from torch import nn
from torch import nn
from typing import Dict, List, Tuple

from .rec import ConversationModule
    
class SelfAttentionBatch(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionBatch, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        return

    def forward(self, h):
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)
        attention = F.sigmoid(e)
        return torch.matmul(attention, h)
    
class KBRDModel(torch.nn.Module):
    def __init__(
            self, n_item:int, n_entity:int, n_word:int,
            model_config:Dict, device:str
        ) -> None:
        super(KBRDModel, self).__init__()
        self.device = device
        self.pad_token_idx = 0
        self.n_word = n_word
        
        self.vocab_size = n_item
        self.token_emb_dim = model_config["emb_dim"]
        
        self.n_entity = n_entity
        self.num_bases = model_config["num_bases"]
        self.kg_emb_dim = model_config["emb_dim"]
        self.user_emb_dim = self.kg_emb_dim
        self.build_model()
        return


    def build_model(self):
        self._build_embedding()
        self._build_kg_layer()
        self._build_recommendation_layer()
        self.con_module = ConversationModule(self.user_emb_dim, self.n_word, self.device)
        return

    def _build_embedding(self):
        self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)
        return

    def _build_kg_layer(self):
        self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
        return

    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()
        return

    def encode_user(self, entity_lists, kg_embedding):
        user_repr_list = []
        for entity_list in entity_lists:
            if len(entity_list) == 0:
                user_repr_list.append(torch.zeros(self.user_emb_dim, device=self.device))
                continue
            user_repr = kg_embedding[entity_list]
            user_repr = self.kg_attn(user_repr)
            user_repr_list.append(user_repr)
        return torch.stack(user_repr_list, dim=0)  # (bs, dim)

    def rec_forward(
            self, batch_data:List[Dict],
            item_edger:Dict, entity_edger:Dict, word_edger:Dict
        ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        self.item_adj = item_edger
        self.entity_adj = entity_edger
        self.word_adj = word_edger

        related_item = [meta_data["item"] for meta_data in batch_data]
        related_entity = [meta_data["entity"] for meta_data in batch_data]
        related_word = [meta_data["word"] for meta_data in batch_data]
        labels = torch.LongTensor([meta_data["label"] for meta_data in batch_data]).to(self.device)
        token_embedding = self.token_embedding.weight

        user_embedding = self.encode_user(
            related_entity,
            token_embedding,
        )

        logits = F.linear(user_embedding, token_embedding, self.rec_bias.bias)
        loss = self.rec_loss(logits, labels)
        return logits, labels, loss

    def con_forward(
            self, batch_data:List[Dict],
            item_edger:Dict, entity_edger:Dict, word_edger:Dict
        ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        self.item_adj = item_edger
        self.entity_adj = entity_edger
        self.word_adj = word_edger

        related_item = [meta_data["item"] for meta_data in batch_data]
        related_entity = [meta_data["entity"] for meta_data in batch_data]
        related_words = [meta_data["word"] for meta_data in batch_data]
        labels = torch.LongTensor([meta_data["label"] for meta_data in batch_data]).to(self.device)
        texts = [meta_data["text"][1:] for meta_data in batch_data]
        token_embedding = self.token_embedding.weight

        user_embeddings = self.encode_user(
            related_entity,
            token_embedding,
        )

        logits = []
        labels = []
        for related_word, user_embedding, label in zip(related_words, user_embeddings, texts):
            label = torch.LongTensor(label).to(self.device)
            logit = self.con_module.decode_forced(related_word, user_embedding)[:label.size(0)]
            logits.append(logit)
            labels.append(label)

        labels = torch.concatenate(labels, dim=0)
        logits = torch.concatenate(logits, dim=0)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        response = torch.argmax(logits, dim=-1)

        return response, labels, loss