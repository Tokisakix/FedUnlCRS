from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

from .rec import ConversationModule

class MHItemAttention(torch.nn.Module):
    def __init__(self, dim, head_num):
        super().__init__()
        self.MHA = torch.nn.MultiheadAttention(dim, head_num, batch_first=True)
        return

    def forward(self, related_entity, context_entity):
        context_entity = torch.unsqueeze(context_entity, 0)
        related_entity = torch.unsqueeze(related_entity, 0)
        output, _ = self.MHA(context_entity, related_entity, related_entity)
        return torch.squeeze(output, 0)

class SelfAttentionBatch(torch.nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionBatch, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = torch.nn.Parameter(torch.zeros(size=(self.dim, self.da)), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros(size=(self.da, 1)), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.b.data, gain=1.414)
        return

    def forward(self, h):
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e, dim=0)
        return torch.matmul(attention, h)

class HyCoRec(torch.nn.Module):
    def __init__(
            self, n_item:int, n_entity:int, n_word:int,
            model_config:Dict, device:str
        ) -> None:
        super().__init__()

        self.n_item = n_item
        self.n_entity = n_entity
        self.n_word = n_word
        self.device = device

        # kg
        self.num_bases = model_config["num_bases"]
        self.kg_emb_dim = model_config["emb_dim"]
        self.user_emb_dim = model_config["emb_dim"]

        # pooling
        self.pooling = model_config["pooling_methon"]
        self.mha_n_heads = model_config["mha_n_heads"]
        self.build_model()
        return

    def build_model(self):
        self._build_embedding()
        self._build_kg_layer()
        self._build_recommendation_layer()
        self.con_module = ConversationModule(self.user_emb_dim, self.n_word, self.device)
        return

    def _build_embedding(self):
        self.item_embedding = torch.nn.Embedding(self.n_item, self.kg_emb_dim, 0)
        torch.nn.init.normal_(self.item_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        torch.nn.init.constant_(self.item_embedding.weight[0], 0)

        self.entity_embedding = torch.nn.Embedding(self.n_entity, self.kg_emb_dim, 0)
        torch.nn.init.normal_(self.entity_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        torch.nn.init.constant_(self.entity_embedding.weight[0], 0)

        self.word_embedding = torch.nn.Embedding(self.n_word, self.kg_emb_dim, 0)
        torch.nn.init.normal_(self.word_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        torch.nn.init.constant_(self.word_embedding.weight[0], 0)
        return

    def _build_kg_layer(self):
        # hypergraph convolution
        self.hyper_conv_item = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
        self.hyper_conv_entity = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
        self.hyper_conv_word = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)

        # attention type
        self.item_attn = MHItemAttention(self.kg_emb_dim, self.mha_n_heads)

        # pooling
        if self.pooling == "Attn":
            self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
            self.kg_attn_his = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
        return

    def _build_recommendation_layer(self):
        self.rec_bias = torch.nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = torch.nn.CrossEntropyLoss()
        return

    def _get_hypergraph(self, related, adj):
        related_items_set = set()
        for related_items in related:
            related_items_set.add(related_items)
        session_related_items = list(related_items_set)

        hypergraph_nodes, hypergraph_edges, hyper_edge_counter = list(), list(), 0
        for item in session_related_items:
            hypergraph_nodes.append(item)
            hypergraph_edges.append(hyper_edge_counter)
            neighbors = adj[item]
            hypergraph_nodes += neighbors
            hypergraph_edges += [hyper_edge_counter] * len(neighbors)
            hyper_edge_counter += 1
        hyper_edge_index = torch.tensor([hypergraph_nodes, hypergraph_edges], device=self.device)
        return list(set(hypergraph_nodes)), hyper_edge_index

    def _get_embedding(self, hypergraph_items, embedding, tot_sub, adj):
        knowledge_embedding_list = []
        for item in hypergraph_items:
            sub_graph = [item] + list(adj[item])
            sub_graph = [tot_sub[item] for item in sub_graph]
            sub_graph_embedding = embedding[sub_graph]
            sub_graph_embedding = torch.mean(sub_graph_embedding, dim=0)
            knowledge_embedding_list.append(sub_graph_embedding)
        res_embedding = torch.zeros(1, self.kg_emb_dim).to(self.device)
        if len(knowledge_embedding_list) > 0:
            res_embedding = torch.stack(knowledge_embedding_list, dim=0)
        return res_embedding

    @staticmethod
    def flatten(inputs):
        outputs = set()
        for li in inputs:
            for i in li:
                outputs.add(i)
        return list(outputs)

    def _attention_and_gating(self, session_embedding, knowledge_embedding, conceptnet_embedding, context_embedding):
        related_embedding = torch.cat((session_embedding, knowledge_embedding, conceptnet_embedding), dim=0)
        if context_embedding is None:
            if self.pooling == "Attn":
                user_repr = self.kg_attn_his(related_embedding)
            else:
                assert self.pooling == "Mean"
                user_repr = torch.mean(related_embedding, dim=0)
        elif self.pooling == "Attn":
            attentive_related_embedding = self.item_attn(related_embedding, context_embedding)
            user_repr = self.kg_attn_his(attentive_related_embedding)
            user_repr = torch.unsqueeze(user_repr, dim=0)
            user_repr = torch.cat((context_embedding, user_repr), dim=0)
            user_repr = self.kg_attn(user_repr)
        else:
            assert self.pooling == "Mean"
            attentive_related_embedding = self.item_attn(related_embedding, context_embedding)
            user_repr = torch.mean(attentive_related_embedding, dim=0)
            user_repr = torch.unsqueeze(user_repr, dim=0)
            user_repr = torch.cat((context_embedding, user_repr), dim=0)
            user_repr = torch.mean(user_repr, dim=0)
        return user_repr

    def _get_hllm_embedding(self, tot_embedding, hllm_hyper_graph, adj, conv):
        hllm_hyper_edge_A = []
        hllm_hyper_edge_B = []
        for idx, hyper_edge in enumerate(hllm_hyper_graph):
            hllm_hyper_edge_A += [item for item in hyper_edge]
            hllm_hyper_edge_B += [idx] * len(hyper_edge)

        hllm_items = list(set(hllm_hyper_edge_A))
        sub_item2id = {item:idx for idx, item in enumerate(hllm_items)}
        sub_embedding = tot_embedding[hllm_items]

        hllm_hyper_edge = [[sub_item2id[item] for item in hllm_hyper_edge_A], hllm_hyper_edge_B]
        hllm_hyper_edge = torch.LongTensor(hllm_hyper_edge).to(self.device)

        embedding = conv(sub_embedding, hllm_hyper_edge)

        return embedding
    
    def encode_user_repr(self, related_items, related_entities, related_words, tot_item_embedding, tot_entity_embedding, tot_word_embedding):
        if len(related_items) or len(related_words) == 0:
            if len(related_entities) == 0:
                user_repr = torch.zeros(self.user_emb_dim, device=self.device)
            elif self.pooling == "Attn":
                user_repr = tot_entity_embedding[related_entities]
                user_repr = self.kg_attn(user_repr)
            else:
                assert self.pooling == "Mean"
                user_repr = tot_entity_embedding[related_entities]
                user_repr = torch.mean(user_repr, dim=0)
            return user_repr

        item_embedding = torch.zeros((1, self.kg_emb_dim), device=self.device)
        if len(related_items) > 0:
            items, item_hyper_edge_index = self._get_hypergraph(related_items, self.item_adj)
            sub_item_embedding, sub_item_edge_index, item_tot2sub = self._before_hyperconv(tot_item_embedding, items, item_hyper_edge_index, self.item_adj)
            raw_item_embedding = self.hyper_conv_item(sub_item_embedding, sub_item_edge_index)
            item_embedding = raw_item_embedding
            # item_embedding = self._get_embedding(items, raw_item_embedding, item_tot2sub, self.item_adj)

        entity_embedding = torch.zeros((1, self.kg_emb_dim), device=self.device)
        if len(related_entities) > 0:
            entities, entity_hyper_edge_index = self._get_hypergraph(related_entities, self.entity_adj)
            sub_entity_embedding, sub_entity_edge_index, entity_tot2sub = self._before_hyperconv(tot_entity_embedding, entities, entity_hyper_edge_index, self.entity_adj)
            raw_entity_embedding = self.hyper_conv_entity(sub_entity_embedding, sub_entity_edge_index)
            entity_embedding = raw_entity_embedding
            # entity_embedding = self._get_embedding(entities, raw_entity_embedding, entity_tot2sub, self.entity_adj)
            
        word_embedding = torch.zeros((1, self.kg_emb_dim), device=self.device)
        if len(related_words) > 0:
            owrds, word_hyper_edge_index = self._get_hypergraph(related_words, self.word_adj)
            sub_word_embedding, sub_word_edge_index, word_tot2sub = self._before_hyperconv(tot_word_embedding, owrds, word_hyper_edge_index, self.word_adj)
            raw_word_embedding = self.hyper_conv_word(sub_word_embedding, sub_word_edge_index)
            word_embedding = raw_word_embedding
            # word_embedding = self._get_embedding(owrds, raw_word_embedding, word_tot2sub, self.word_adj)

        if len(related_entities) == 0:
            user_repr = self._attention_and_gating(item_embedding, entity_embedding, word_embedding, None)
        else:
            context_embedding = tot_entity_embedding[related_entities]
            user_repr = self._attention_and_gating(item_embedding, entity_embedding, word_embedding, context_embedding)
        return user_repr

    def encode_user(self, batch_related_items, batch_related_entities, batch_related_words, tot_item_embedding, tot_entity_embedding, tot_word_embedding):
        user_repr_list = []
        for related_items, related_entities, related_words in zip(batch_related_items, batch_related_entities, batch_related_words):
            user_repr = self.encode_user_repr(related_items, related_entities, related_words, tot_item_embedding, tot_entity_embedding, tot_word_embedding)
            user_repr_list.append(user_repr)
        user_embedding = torch.stack(user_repr_list, dim=0)
        return user_embedding

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
        item_embedding = self.entity_embedding.weight
        entity_embedding = self.entity_embedding.weight
        token_embedding = self.word_embedding.weight

        user_embedding = self.encode_user(
            related_item,
            related_entity,
            related_word,
            item_embedding,
            entity_embedding,
            token_embedding,
        )

        logits = F.linear(user_embedding, entity_embedding, self.rec_bias.bias)
        loss = self.rec_loss(logits, labels)
        return logits, labels, loss

    def _before_hyperconv(self, embeddings:torch.FloatTensor, hypergraph_items:List[int], edge_index:torch.LongTensor, adj):
        sub_items = []
        edge_index = edge_index.cpu().numpy()
        for item in hypergraph_items:
            sub_items += [item] + list(adj[item])
        sub_items = list(set(sub_items))
        tot2sub = {tot:sub for sub, tot in enumerate(sub_items)}
        sub_embeddings = embeddings[sub_items]
        edge_index = [[tot2sub[v] for v in edge_index[0]], list(edge_index[1])]
        sub_edge_index = torch.tensor(edge_index).long()
        sub_edge_index = sub_edge_index.to(self.device)
        return sub_embeddings, sub_edge_index, tot2sub

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
        texts = [meta_data["text"][1:] for meta_data in batch_data]
        item_embedding = self.entity_embedding.weight
        entity_embedding = self.entity_embedding.weight
        token_embedding = self.word_embedding.weight

        user_embeddings = self.encode_user(
            related_item,
            related_entity,
            related_words,
            item_embedding,
            entity_embedding,
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