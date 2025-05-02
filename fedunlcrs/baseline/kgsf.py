import os
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch_geometric.nn import GCNConv, RGCNConv
from typing import Dict, List

def edge_to_pyg_format(edge, type='RGCN'):
    if type == 'RGCN':
        edge_sets = torch.as_tensor(edge, dtype=torch.long)
        edge_idx = edge_sets[:, :2].t()
        edge_type = edge_sets[:, 2]
        return edge_idx, edge_type
    elif type == 'GCN':
        edge_set = [[co[0] for co in edge], [co[1] for co in edge]]
        return torch.as_tensor(edge_set, dtype=torch.long)
    else:
        raise NotImplementedError('type {} has not been implemented', type)
    
class GateLayer(nn.Module):
    def __init__(self, input_dim):
        super(GateLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 2, input_dim)
        self._norm_layer2 = nn.Linear(input_dim, 1)

    def forward(self, input1, input2):
        norm_input = self._norm_layer1(torch.cat([input1, input2], dim=-1))
        gate = torch.sigmoid(self._norm_layer2(norm_input))  # (bs, 1)
        gated_emb = gate * input1 + (1 - gate) * input2  # (bs, dim)
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

    def forward(self, h, mask=None, return_logits=False):
        """
        For the padding tokens, its corresponding mask is True
        if mask==[1, 1, 1, ...]
        """
        # h: (batch, seq_len, dim), mask: (batch, seq_len)
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)  # (batch, seq_len, 1)
        if mask is not None:
            full_mask = -1e30 * mask.float()
            batch_mask = torch.sum((mask == False), -1).bool().float().unsqueeze(-1)  # for all padding one, the mask=0
            mask = full_mask * batch_mask
            e += mask.unsqueeze(-1)
        attention = F.softmax(e, dim=1)  # (batch, seq_len, 1)
        # (batch, dim)
        if return_logits:
            return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1), attention.squeeze(-1)
        else:
            return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1)

class KGSFModel(torch.nn.Module):

    def __init__(self, n_item:int, n_entity:int, n_word:int, model_config:Dict, device:str):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        super(KGSFModel, self).__init__()
        self.device = device
        # vocab
        self.vocab_size = n_item
        self.pad_token_idx = 0
        self.token_emb_dim = model_config['emb_dim']
        # kg
        self.n_word = n_word
        self.n_entity = n_entity
        self.pad_word_idx = 0
        self.pad_entity_idx = 0

        self.num_bases = model_config['num_bases']
        self.kg_emb_dim = model_config['emb_dim']
        self.build_model()

    def build_model(self):
        self._init_embeddings()
        self._build_kg_layer()
        self._build_infomax_layer()
        self._build_recommendation_layer()

    def _init_embeddings(self):
        self.entity_kg_embedding = nn.Embedding(self.n_entity, self.token_emb_dim, self.pad_token_idx)
        nn.init.normal_(self.entity_kg_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.entity_kg_embedding.weight[self.pad_token_idx], 0)

        self.word_kg_embedding = nn.Embedding(self.n_word, self.kg_emb_dim, self.pad_word_idx)
        nn.init.normal_(self.word_kg_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.word_kg_embedding.weight[self.pad_word_idx], 0)

        logger.debug('[Finish init embeddings]')

    def _build_kg_layer(self):
        # db encoder
        # self.entity_encoder = RGCNConv(self.n_entity, self.kg_emb_dim, self.n_relation, self.num_bases)
        self.entity_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)

        # concept encoder
        self.word_encoder = GCNConv(self.kg_emb_dim, self.kg_emb_dim)
        self.word_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)

        # gate mechanism
        self.gate_layer = GateLayer(self.kg_emb_dim)

        logger.debug('[Finish build kg layer]')

    def _build_infomax_layer(self):
        self.infomax_norm = nn.Linear(self.kg_emb_dim, self.kg_emb_dim)
        self.infomax_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.infomax_loss = nn.MSELoss(reduction='sum')

        logger.debug('[Finish build infomax layer]')

    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()

        logger.debug('[Finish build rec layer]')

    def rec_forward(self, batch_data:List[Dict], item_edger:Dict, entity_edger:Dict, word_edger:Dict):
        related_entity = [meta_data["entity"] for meta_data in batch_data]
        related_word = [meta_data["word"] for meta_data in batch_data]
        labels = torch.LongTensor([meta_data["label"] for meta_data in batch_data]).to(self.device)

        # entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        # word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)
        entity_graph_representations = self.entity_kg_embedding.weight
        word_graph_representations = self.word_kg_embedding.weight

        user_embedding = []
        for entity_list, word_list in zip(related_entity, related_word):
            entity_representations = entity_graph_representations[entity_list]
            word_representations = word_graph_representations[word_list]

            # entity_attn_rep = self.entity_self_attn(entity_representations)
            # word_attn_rep = self.word_self_attn(word_representations)
            entity_attn_rep = entity_representations.mean(dim=0, keepdim=True)
            word_attn_rep = word_representations.mean(dim=0, keepdim=True)

            user_rep = self.gate_layer(entity_attn_rep, word_attn_rep)
            user_embedding.append(user_rep)
        user_embedding = torch.concatenate(user_embedding, dim=0)
        rec_scores = F.linear(user_embedding, entity_graph_representations, self.rec_bias.bias)  # (bs, #entity)

        rec_loss = self.rec_loss(rec_scores, labels)

        info_loss_mask = torch.sum(labels)
        if info_loss_mask.item() == 0:
            info_loss = None
        elif False:
            word_info_rep = self.infomax_norm(word_attn_rep)  # (bs, dim)
            info_predict = F.linear(word_info_rep, entity_graph_representations,
                                    self.infomax_bias.bias)  # (bs, #entity)
            info_loss = self.infomax_loss(info_predict, labels) / info_loss_mask

        return rec_scores, None, rec_loss
