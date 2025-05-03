from typing import Dict, List

import torch
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

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

class MHIMModel(torch.nn.Module):
    def __init__(self, n_item:int, n_entity:int, n_word:int, model_config:Dict, device:str):
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
        self.pooling = "Mean"
        self.mha_n_heads = model_config["mha_n_heads"]
        self.build_model()
        return

    def build_model(self):
        self._build_embedding()
        self._build_kg_layer()
        self._build_recommendation_layer()
        return

    def _build_embedding(self):
        self.item_embedding = torch.nn.Embedding(self.n_item, self.kg_emb_dim, 0)
        torch.nn.init.normal_(self.item_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        torch.nn.init.constant_(self.item_embedding.weight[0], 0)

        self.entity_embedding = torch.nn.Embedding(self.n_entity, self.kg_emb_dim, 0)
        torch.nn.init.normal_(self.entity_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        torch.nn.init.constant_(self.entity_embedding.weight[0], 0)
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

    def _build_conversation_layer(self):
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))
        self.entity_to_token = nn.Linear(self.kg_emb_dim, self.token_emb_dim, bias=True)
        self.related_encoder = TransformerEncoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.pad_token_idx,
            self.learn_positional_embeddings,
            self.embeddings_scale,
            self.reduction,
            self.n_positions
        )
        self.context_encoder = TransformerEncoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.pad_token_idx,
            self.learn_positional_embeddings,
            self.embeddings_scale,
            self.reduction,
            self.n_positions
        )
        self.decoder = TransformerDecoderKG(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.embeddings_scale,
            self.learn_positional_embeddings,
            self.pad_token_idx,
            self.n_positions
        )
        self.user_proj_1 = nn.Linear(self.user_emb_dim, self.user_proj_dim)
        self.user_proj_2 = nn.Linear(self.user_proj_dim, self.vocab_size)
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)

        self.copy_proj_1 = nn.Linear(2 * self.token_emb_dim, self.token_emb_dim)
        self.copy_proj_2 = nn.Linear(self.token_emb_dim, self.vocab_size)
        logger.debug('[Build conversation layer]')

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

    def _attention_and_gating(self, session_embedding, knowledge_embedding, context_embedding):
        related_embedding = torch.cat((session_embedding, knowledge_embedding), dim=0)
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

        if len(related_entities) == 0:
            user_repr = self._attention_and_gating(item_embedding, entity_embedding, None)
        else:
            context_embedding = tot_entity_embedding[related_entities]
            user_repr = self._attention_and_gating(item_embedding, entity_embedding, context_embedding)
        return user_repr

    def encode_user(self, batch_related_items, batch_related_entities, batch_related_words, tot_item_embedding, tot_entity_embedding, tot_word_embedding):
        user_repr_list = []
        for related_items, related_entities, related_words in zip(batch_related_items, batch_related_entities, batch_related_words):
            user_repr = self.encode_user_repr(related_items, related_entities, related_words, tot_item_embedding, tot_entity_embedding, tot_word_embedding)
            user_repr_list.append(user_repr)
        user_embedding = torch.stack(user_repr_list, dim=0)
        return user_embedding

    def rec_forward(self, batch_data:List[Dict], item_edger:Dict, entity_edger:Dict, word_edger:Dict) -> torch.FloatTensor:
        self.item_adj = item_edger
        self.entity_adj = entity_edger
        self.word_adj = word_edger

        related_item = [meta_data["item"] for meta_data in batch_data]
        related_entity = [meta_data["entity"] for meta_data in batch_data]
        related_word = [meta_data["word"] for meta_data in batch_data]
        labels = torch.LongTensor([meta_data["label"] for meta_data in batch_data]).to(self.device)
        item_embedding = self.entity_embedding.weight
        entity_embedding = self.entity_embedding.weight
        token_embedding = self.entity_embedding.weight

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

    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)

    def freeze_parameters(self):
        freeze_models = [
            self.kg_embedding,
            self.kg_encoder,
            self.hyper_conv_session,
            self.hyper_conv_knowledge,
            self.item_attn,
            self.rec_bias
        ]
        if self.pooling == "Attn":
            freeze_models.append(self.kg_attn)
            freeze_models.append(self.kg_attn_his)
        for model in freeze_models:
            for p in model.parameters():
                p.requires_grad = False

    def encode_session(self, batch_related_items, batch_context_entities, kg_embedding):
        """
            Return: session_repr (batch_size, batch_seq_len, token_emb_dim), mask (batch_size, batch_seq_len)
        """
        session_repr_list = []
        for session_related_items, context_entities in zip(batch_related_items, batch_context_entities):
            flattened_session_related_items = self.flatten(session_related_items)

            # COLD START
            if len(flattened_session_related_items) == 0:
                if len(context_entities) == 0:
                    session_repr_list.append(None)
                else:
                    session_repr = kg_embedding[context_entities]
                    session_repr_list.append(session_repr)
                continue

            # TOTAL
            hypergraph_items, session_hyper_edge_index = self._get_session_hypergraph(session_related_items)
            session_embedding = self.hyper_conv_session(kg_embedding, session_hyper_edge_index)
            session_embedding = session_embedding[hypergraph_items]
            _, knowledge_hyper_edge_index = self._get_knowledge_hypergraph(session_related_items)
            raw_knowledge_embedding = self.hyper_conv_knowledge(kg_embedding, knowledge_hyper_edge_index)
            knowledge_embedding = self._get_knowledge_embedding(hypergraph_items, raw_knowledge_embedding)
            if len(context_entities) == 0:
                session_repr = torch.cat((session_embedding, knowledge_embedding), dim=0)
                session_repr_list.append(session_repr)
            else:
                context_embedding = kg_embedding[context_entities]
                session_repr = torch.cat((session_embedding, knowledge_embedding, context_embedding), dim=0)
                session_repr_list.append(session_repr)

        batch_seq_len = max([session_repr.size(0) for session_repr in session_repr_list if session_repr is not None])
        mask_list = []
        for i in range(len(session_repr_list)):
            if session_repr_list[i] is None:
                mask_list.append([False] * batch_seq_len)
                zero_repr = torch.zeros((batch_seq_len, self.kg_emb_dim), device=self.device, dtype=torch.float)
                session_repr_list[i] = zero_repr
            else:
                mask_list.append([False] * (batch_seq_len - session_repr_list[i].size(0)) + [True] * session_repr_list[i].size(0))
                zero_repr = torch.zeros((batch_seq_len - session_repr_list[i].size(0), self.kg_emb_dim),
                                        device=self.device, dtype=torch.float)
                session_repr_list[i] = torch.cat((zero_repr, session_repr_list[i]), dim=0)

        session_repr_embedding = torch.stack(session_repr_list, dim=0)
        session_repr_embedding = self.entity_to_token(session_repr_embedding)
        return session_repr_embedding, torch.tensor(mask_list, device=self.device, dtype=torch.bool)

    def decode_forced(self, related_encoder_state, context_encoder_state, session_state, user_embedding, resp):
        bsz = resp.size(0)
        seqlen = resp.size(1)
        inputs = resp.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, related_encoder_state, context_encoder_state, session_state)
        token_logits = F.linear(latent, self.token_embedding.weight)
        user_logits = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)

        user_latent = self.entity_to_token(user_embedding)
        user_latent = user_latent.unsqueeze(1).expand(-1, seqlen, -1)
        copy_latent = torch.cat((user_latent, latent), dim=-1)
        copy_logits = self.copy_proj_2(torch.relu(self.copy_proj_1(copy_latent)))
        if self.dataset == 'HReDial':
            copy_logits = copy_logits * self.copy_mask.unsqueeze(0).unsqueeze(0)  # not for tg-redial
        sum_logits = token_logits + user_logits + copy_logits
        _, preds = sum_logits.max(dim=-1)
        return sum_logits, preds

    def decode_greedy(self, related_encoder_state, context_encoder_state, session_state, user_embedding):
        bsz = context_encoder_state[0].shape[0]
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for i in range(self.longest_label):
            scores, incr_state = self.decoder(xs, related_encoder_state, context_encoder_state, session_state, incr_state)  # incr_state is always None
            scores = scores[:, -1:, :]
            token_logits = F.linear(scores, self.token_embedding.weight)
            user_logits = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)

            user_latent = self.entity_to_token(user_embedding)
            user_latent = user_latent.unsqueeze(1).expand(-1, 1, -1)
            copy_latent = torch.cat((user_latent, scores), dim=-1)
            copy_logits = self.copy_proj_2(torch.relu(self.copy_proj_1(copy_latent)))
            if self.dataset == 'HReDial':
                copy_logits = copy_logits * self.copy_mask.unsqueeze(0).unsqueeze(0)  # not for tg-redial
            sum_logits = token_logits + user_logits + copy_logits
            probs, preds = sum_logits.max(dim=-1)
            logits.append(scores)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.end_token_idx).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def converse(self, batch, mode):
        related_tokens = batch['related_tokens']
        context_tokens = batch['context_tokens']
        related_items = batch['related_items']
        related_entities = batch['related_entities']
        context_entities = batch['context_entities']
        response = batch['response']
        kg_embedding = self.kg_encoder(self.kg_embedding.weight, self.edge_idx, self.edge_type)  # (n_entity, emb_dim)
        session_state = self.encode_session(
            related_items,
            context_entities,
            kg_embedding
        )  # (batch_size, batch_seq_len, token_emb_dim)
        user_embedding = self.encode_user(
            related_entities,
            related_items,
            context_entities,
            kg_embedding
        )  # (batch_size, emb_dim)
        related_encoder_state = self.related_encoder(related_tokens)
        context_encoder_state = self.context_encoder(context_tokens)
        if mode != 'test':
            self.longest_label = max(self.longest_label, response.shape[1])
            logits, preds = self.decode_forced(related_encoder_state, context_encoder_state, session_state, user_embedding, response)
            logits = logits.view(-1, logits.shape[-1])
            labels = response.view(-1)
            return self.conv_loss(logits, labels), preds
        else:
            _, preds = self.decode_greedy(related_encoder_state, context_encoder_state, session_state, user_embedding)
            return preds