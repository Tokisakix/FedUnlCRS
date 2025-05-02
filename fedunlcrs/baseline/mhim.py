import json
import os.path
import random
from typing import Dict, List
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from tqdm import tqdm
from torch_geometric.nn import RGCNConv, HypergraphConv

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

    def forward(self, h):
        # h: (N, dim)
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e, dim=0)  # (N)
        return torch.matmul(attention, h)  # (dim)

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


class MHItemAttention(nn.Module):
    def __init__(self, dim, head_num):
        super(MHItemAttention, self).__init__()
        self.MHA = torch.nn.MultiheadAttention(dim, head_num, batch_first=True)

    def forward(self, related_entity, context_entity):
        """
            input:
                related_entity: (n_r, dim)
                context_entity: (n_c, dim)
            output:
                related_context_entity: (n_c, dim)
        """
        context_entity = torch.unsqueeze(context_entity, 0)
        related_entity = torch.unsqueeze(related_entity, 0)
        output, _ = self.MHA(context_entity, related_entity, related_entity)
        return torch.squeeze(output, 0)


class MHIMModel(torch.nn.Module):
    def __init__(self, n_item:int, n_entity:int, n_word:int, model_config:Dict, device:str, processed_entity_kg):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        super(MHIMModel, self).__init__()
        self.device = device
        # vocab
        self.pad_token_idx = 0
        self.start_token_idx = 0
        self.end_token_idx =0
        self.vocab_size = n_item
        self.token_emb_dim = model_config["emb_dim"]
        self.dataset_name = model_config["dataset_name"]
        # kg
        self.n_entity = n_entity
        self.n_relation = processed_entity_kg['n_relation']
        self.edge_idx, self.edge_type = edge_to_pyg_format(self.entity_kg['edge'], 'RGCN')
        self.edge_idx = self.edge_idx.to(device)
        self.edge_type = self.edge_type.to(device)
        self.num_bases = model_config["num_bases"]
        self.kg_emb_dim = model_config["emb_dim"]
        self.user_emb_dim = model_config["emb_dim"]
        # pooling
        self.pooling = model_config["pooling"]
        assert self.pooling == 'Attn' or self.pooling == 'Mean'
        # MHA
        self.mha_n_heads = model_config["mha_n_heads"]
        self.extension_strategy = model_config["extension_strategy"]
        self.build_model()

    def build_model(self, *args, **kwargs):
        self._build_copy_mask()
        self._build_adjacent_matrix()
        self._build_embedding()
        self._build_kg_layer()
        self._build_recommendation_layer()

    def _build_copy_mask(self):
        token_filename = json.load(open(os.path.join("data", self.dataset_name, "token2id.json"), "r", encoding="utf-8"))
        token_file = open(token_filename, 'r')
        token2id = json.load(token_file)
        id2token = {token2id[token]: token for token in token2id}
        self.copy_mask = list()
        for i in range(len(id2token)):
            token = id2token[i]
            if token[0] == '@':
                self.copy_mask.append(True)
            else:
                self.copy_mask.append(False)
        self.copy_mask = torch.as_tensor(self.copy_mask).to(self.device)

    def _build_adjacent_matrix(self):
        graph = dict()
        for head, tail, relation in tqdm(self.entity_kg['edge']):
            graph[head] = graph.get(head, []) + [tail]
        adj = dict()
        for entity in tqdm(range(self.n_entity)):
            adj[entity] = set()
            if entity not in graph:
                continue
            last_hop = {entity}
            for _ in range(1):
                buffer = set()
                for source in last_hop:
                    adj[entity].update(graph[source])
                    buffer.update(graph[source])
                last_hop = buffer
        self.adj = adj
        logger.info(f"[Adjacent Matrix built.]")

    def _build_embedding(self):
        self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)
        self.kg_embedding = nn.Embedding(self.n_entity, self.kg_emb_dim, 0)
        nn.init.normal_(self.kg_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.kg_embedding.weight[0], 0)
        logger.debug('[Build embedding]')

    def _build_kg_layer(self):
        # graph encoder
        self.kg_encoder = RGCNConv(self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        # hypergraph convolution
        self.hyper_conv_session = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
        self.hyper_conv_knowledge = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
        # attention type
        self.item_attn = MHItemAttention(self.kg_emb_dim, self.mha_n_heads)
        # pooling
        if self.pooling == 'Attn':
            self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
            self.kg_attn_his = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
        logger.debug('[Build kg layer]')

    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()
        logger.debug('[Build recommendation layer]')

    def _get_session_hypergraph(self, session_related_entities):
        hypergraph_nodes, hypergraph_edges, hyper_edge_counter = list(), list(), 0
        for related_entities in session_related_entities:
            if len(related_entities) == 0:
                continue
            hypergraph_nodes += related_entities
            hypergraph_edges += [hyper_edge_counter] * len(related_entities)
            hyper_edge_counter += 1
        hyper_edge_index = torch.tensor([hypergraph_nodes, hypergraph_edges], device=self.device)
        return list(set(hypergraph_nodes)), hyper_edge_index

    def _get_knowledge_hypergraph(self, session_related_items):
        related_items_set = set()
        for related_items in session_related_items:
            related_items_set.update(related_items)
        session_related_items = list(related_items_set)

        hypergraph_nodes, hypergraph_edges, hyper_edge_counter = list(), list(), 0
        for item in session_related_items:
            hypergraph_nodes.append(item)
            hypergraph_edges.append(hyper_edge_counter)
            neighbors = list(self.adj[item])
            hypergraph_nodes += neighbors
            hypergraph_edges += [hyper_edge_counter] * len(neighbors)
            hyper_edge_counter += 1
        hyper_edge_index = torch.tensor([hypergraph_nodes, hypergraph_edges], device=self.device)
        return list(set(hypergraph_nodes)), hyper_edge_index

    def _get_knowledge_embedding(self, hypergraph_items, raw_knowledge_embedding):
        knowledge_embedding_list = []
        for item in hypergraph_items:
            sub_graph = [item] + list(self.adj[item])
            sub_graph_embedding = raw_knowledge_embedding[sub_graph]
            sub_graph_embedding = torch.mean(sub_graph_embedding, dim=0)
            knowledge_embedding_list.append(sub_graph_embedding)
        return torch.stack(knowledge_embedding_list, dim=0)

    @staticmethod
    def flatten(inputs):
        outputs = set()
        for li in inputs:
            for i in li:
                outputs.add(i)
        return list(outputs)

    def _attention_and_gating(self, session_embedding, knowledge_embedding, context_embedding):
        related_embedding = torch.cat((session_embedding, knowledge_embedding), dim=0)
        if context_embedding is None:
            if self.pooling == 'Attn':
                user_repr = self.kg_attn_his(related_embedding)
            else:
                assert self.pooling == 'Mean'
                user_repr = torch.mean(related_embedding, dim=0)
        elif self.pooling == 'Attn':
            attentive_related_embedding = self.item_attn(related_embedding, context_embedding)
            user_repr = self.kg_attn_his(attentive_related_embedding)
            user_repr = torch.unsqueeze(user_repr, dim=0)
            user_repr = torch.cat((context_embedding, user_repr), dim=0)
            user_repr = self.kg_attn(user_repr)
        else:
            assert self.pooling == 'Mean'
            attentive_related_embedding = self.item_attn(related_embedding, context_embedding)
            user_repr = torch.mean(attentive_related_embedding, dim=0)
            user_repr = torch.unsqueeze(user_repr, dim=0)
            user_repr = torch.cat((context_embedding, user_repr), dim=0)
            user_repr = torch.mean(user_repr, dim=0)
        return user_repr

    def encode_user(self, batch_related_entities, batch_related_items, batch_context_entities, kg_embedding):
        user_repr_list = []
        for session_related_entities, session_related_items, context_entities in zip(batch_related_entities, batch_related_items, batch_context_entities):
            flattened_session_related_items = self.flatten(session_related_items)

            # COLD START
            if len(flattened_session_related_items) == 0:
                if len(context_entities) == 0:
                    user_repr = torch.zeros(self.user_emb_dim, device=self.device)
                elif self.pooling == 'Attn':
                    user_repr = kg_embedding[context_entities]
                    user_repr = self.kg_attn(user_repr)
                else:
                    assert self.pooling == 'Mean'
                    user_repr = kg_embedding[context_entities]
                    user_repr = torch.mean(user_repr, dim=0)
                user_repr_list.append(user_repr)
                continue

            hypergraph_items, session_hyper_edge_index = self._get_session_hypergraph(session_related_items)
            session_embedding = self.hyper_conv_session(kg_embedding, session_hyper_edge_index)
            session_embedding = session_embedding[hypergraph_items]
            _, knowledge_hyper_edge_index = self._get_knowledge_hypergraph(session_related_items)
            raw_knowledge_embedding = self.hyper_conv_knowledge(kg_embedding, knowledge_hyper_edge_index)
            knowledge_embedding = self._get_knowledge_embedding(hypergraph_items, raw_knowledge_embedding)
            if len(context_entities) == 0:
                user_repr = self._attention_and_gating(session_embedding, knowledge_embedding, None)
            else:
                context_embedding = kg_embedding[context_entities]
                user_repr = self._attention_and_gating(session_embedding, knowledge_embedding, context_embedding)
            user_repr_list.append(user_repr)
        return torch.stack(user_repr_list, dim=0)

    def rec_recommend(self, batch_data:List[Dict], item_edger:Dict, entity_edger:Dict, word_edger:Dict):
        related_entity = [meta_data["entity"] for meta_data in batch_data]
        related_items = [meta_data["item"] for meta_data in batch_data]
        labels = torch.LongTensor([meta_data["label"] for meta_data in batch_data]).to(self.device)
        context_entities = [meta_data["entity"] for meta_data in batch_data]
        kg_embedding = self.kg_encoder(self.kg_embedding.weight, self.edge_idx, self.edge_type)  # (n_entity, emb_dim)
        extended_items = [meta_data["item"] for meta_data in batch_data]
        for i in range(len(related_items)):
            truncate = min(int(max(2, int(len(related_items[i]) / 4))), len(extended_items[i]))
            if self.extension_strategy == 'Adaptive':
                related_items[i] = related_items[i] + extended_items[i][:truncate]
            else:
                assert self.extension_strategy == 'Random'
                extended_items_sample = random.sample(extended_items[i], truncate)
                related_items[i] = related_items[i] + extended_items_sample

        user_embedding = self.encode_user(
            related_entity,
            related_items,
            context_entities,
            kg_embedding
        )  # (batch_size, emb_dim)
        scores = F.linear(user_embedding, kg_embedding, self.rec_bias.bias)  # (batch_size, n_entity)
        loss = self.rec_loss(scores, labels)
        return scores, scores, loss