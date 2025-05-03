import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from .rec import ConversationModule
    
class GateLayer(nn.Module):
    def __init__(self, input_dim):
        super(GateLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 2, input_dim)
        self._norm_layer2 = nn.Linear(input_dim, 1)
        return

    def forward(self, input1, input2):
        norm_input = self._norm_layer1(torch.cat([input1, input2], dim=-1))
        gate = torch.sigmoid(self._norm_layer2(norm_input))
        gated_emb = gate * input1 + (1 - gate) * input2
        return gated_emb
    
class SelfAttentionSeq(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionSeq, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        return

    def forward(self, h, mask=None, return_logits=False):
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)
        if mask is not None:
            full_mask = -1e30 * mask.float()
            batch_mask = torch.sum((mask == False), -1).bool().float().unsqueeze(-1)
            mask = full_mask * batch_mask
            e += mask.unsqueeze(-1)
        attention = F.softmax(e, dim=1)
        if return_logits:
            return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1), attention.squeeze(-1)
        else:
            return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1)

class NTRDModel(torch.nn.Module):
    def __init__(
            self, n_item:int, n_entity:int, n_word:int,
            model_config:Dict, device:str
        ) -> None:
        super(NTRDModel, self).__init__()
        self.device = device
        
        self.n_word = n_word
        self.vocab_size = n_item
        self.pad_token_idx = 0
        self.token_emb_dim = model_config['emb_dim']
        
        self.n_word = n_word
        self.n_entity = n_entity
        self.pad_word_idx = 0
        self.pad_entity_idx = 0

        self.num_bases = model_config['num_bases']
        self.kg_emb_dim = model_config['emb_dim']
        self.user_emb_dim = model_config['emb_dim']
        self.build_model()
        return
    
    def build_model(self):
        self._init_embeddings()
        self._build_kg_layer()
        self._build_infomax_layer()
        self._build_recommendation_layer()
        self.con_module = ConversationModule(self.user_emb_dim, self.n_word, self.device)
        return
    
    def _init_embeddings(self):
        self.entity_embedding = nn.Embedding(self.n_entity, self.token_emb_dim, self.pad_token_idx)
        nn.init.normal_(self.entity_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.entity_embedding.weight[self.pad_token_idx], 0)
        self.word_kg_embedding = nn.Embedding(self.n_word, self.kg_emb_dim, self.pad_word_idx)
        nn.init.normal_(self.word_kg_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.word_kg_embedding.weight[self.pad_word_idx], 0)
        return

    def _build_kg_layer(self):
        self.entity_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)

        self.word_encoder = GCNConv(self.kg_emb_dim, self.kg_emb_dim)
        self.word_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)

        self.gate_layer = GateLayer(self.kg_emb_dim)
        return

    def _build_infomax_layer(self):
        self.infomax_norm = nn.Linear(self.kg_emb_dim, self.kg_emb_dim)
        self.infomax_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.infomax_loss = nn.MSELoss(reduction='sum')
        return

    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()
        return
    
    def rec_forward(
            self, batch_data:List[Dict],
            item_edger:Dict, entity_edger:Dict, word_edger:Dict
        ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        related_entity = [meta_data["entity"] for meta_data in batch_data]
        related_word = [meta_data["word"] for meta_data in batch_data]
        labels = torch.LongTensor([meta_data["label"] for meta_data in batch_data]).to(self.device)

        max_entity_len = max(len(x) for x in related_entity) if related_entity else 0
        max_word_len = max(len(x) for x in related_word) if related_word else 0

        for i in range(len(related_entity)):
            related_entity[i] += [self.pad_entity_idx] * (max_entity_len - len(related_entity[i]))
        for i in range(len(related_word)):
            related_word[i] += [self.pad_word_idx] * (max_word_len - len(related_word[i]))

        related_entity = torch.LongTensor(related_entity).to(self.device)
        related_word = torch.LongTensor(related_word).to(self.device)

        user_embedding = []
        for entity_list, word_list in zip(related_entity, related_word):
            entity_repr = self.entity_embedding.weight[entity_list]
            entity_representations = entity_repr.mean(dim=0, keepdim=True)

            word_repr = self.word_kg_embedding.weight[word_list]
            word_representations = word_repr.mean(dim=0, keepdim=True)

            user_rep = self.gate_layer(entity_representations, word_representations)
            user_embedding.append(user_rep)
        user_embedding = torch.concatenate(user_embedding, dim=0)
        rec_scores = F.linear(user_embedding, self.entity_embedding.weight, self.rec_bias.bias)

        rec_loss = self.rec_loss(rec_scores, labels)

        info_loss_mask = torch.sum(labels)
        if info_loss_mask.item() == 0:
            info_loss = None
        elif False:
            word_info_rep = self.infomax_norm(word_attn_rep)  
            info_predict = F.linear(word_info_rep, entity_repr, self.infomax_bias.bias)
            labels_one_hot = F.one_hot(labels, num_classes=info_predict.size(1)).float() 
            info_loss = self.infomax_loss(info_predict, labels_one_hot) / info_loss_mask

        return rec_scores, 0.0, rec_loss
    
    def con_forward(
            self, batch_data:List[Dict],
            item_edger:Dict, entity_edger:Dict, word_edger:Dict
        ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        related_entity = [meta_data["entity"] for meta_data in batch_data]
        related_word = [meta_data["word"] for meta_data in batch_data]
        texts = [meta_data["text"][1:] for meta_data in batch_data]
        labels = torch.LongTensor([meta_data["label"] for meta_data in batch_data]).to(self.device)

        max_entity_len = max(len(x) for x in related_entity) if related_entity else 0
        max_word_len = max(len(x) for x in related_word) if related_word else 0

        for i in range(len(related_entity)):
            related_entity[i] += [self.pad_entity_idx] * (max_entity_len - len(related_entity[i]))
        for i in range(len(related_word)):
            related_word[i] += [self.pad_word_idx] * (max_word_len - len(related_word[i]))

        related_entity = torch.LongTensor(related_entity).to(self.device)
        related_word = torch.LongTensor(related_word).to(self.device)

        user_embedding = []
        for entity_list, word_list in zip(related_entity, related_word):
            entity_repr = self.entity_embedding.weight[entity_list]
            entity_representations = entity_repr.mean(dim=0, keepdim=True)

            word_repr = self.word_kg_embedding.weight[word_list]
            word_representations = word_repr.mean(dim=0, keepdim=True)

            user_rep = self.gate_layer(entity_representations, word_representations)
            user_embedding.append(user_rep)
        user_embeddings = torch.concatenate(user_embedding, dim=0)

        logits = []
        labels = []
        related_words = related_word.tolist()
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