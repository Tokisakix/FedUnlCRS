import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch_geometric.nn import RGCNConv
from torch import nn
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
    
class KBRDModel(torch.nn.Module):

    def __init__(self, n_item:int, n_entity:int, n_word:int, model_config:Dict, device:str):
        super(KBRDModel, self).__init__()
        self.device = device
        # vocab
        self.pad_token_idx = 0
        # self.start_token_idx = vocab['start']
        # self.end_token_idx = vocab['end']
        self.vocab_size = n_item
        self.token_emb_dim = model_config["emb_dim"] #dim
        # kg
        self.n_entity = n_entity
        self.num_bases = model_config["num_bases"]
        self.kg_emb_dim = model_config["emb_dim"]
        self.user_emb_dim = self.kg_emb_dim
        self.build_model()


    def build_model(self, *args, **kwargs):
        self._build_embedding()
        self._build_kg_layer()
        self._build_recommendation_layer()

    def _build_embedding(self):
        self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)
        return

    def _build_kg_layer(self):
        # self.kg_encoder = RGCNConv(self.n_entity, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
        logger.debug('[Build kg layer]')

    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()
        logger.debug('[Build recommendation layer]')

     def _build_conversation_layer(self):
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))
        self.dialog_encoder = TransformerEncoder(
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
        self.decoder = TransformerDecoder(
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
        logger.debug('[Build conversation layer]')

    def encode_user(self, entity_lists, kg_embedding):
        user_repr_list = []
        for entity_list in entity_lists:
            if entity_list is None:
                user_repr_list.append(torch.zeros(self.user_emb_dim, device=self.device))
                continue
            user_repr = kg_embedding[entity_list]
            user_repr = self.kg_attn(user_repr)
            user_repr_list.append(user_repr)
        return torch.stack(user_repr_list, dim=0)  # (bs, dim)

    def rec_forward(self, batch_data:List[Dict], item_edger:Dict, entity_edger:Dict, word_edger:Dict) -> torch.FloatTensor:
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

    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)

    def decode_forced(self, encoder_states, user_embedding, resp):
        bsz = resp.size(0)
        seqlen = resp.size(1)
        inputs = resp.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, encoder_states)
        token_logits = F.linear(latent, self.token_embedding.weight)
        user_logits = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)
        sum_logits = token_logits + user_logits
        _, preds = sum_logits.max(dim=-1)
        return sum_logits, preds

    def decode_greedy(self, encoder_states, user_embedding):

        bsz = encoder_states[0].shape[0]
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for i in range(self.longest_label):
            scores, incr_state = self.decoder(xs, encoder_states, incr_state)  # incr_state is always None
            scores = scores[:, -1:, :]
            token_logits = F.linear(scores, self.token_embedding.weight)
            user_logits = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)
            sum_logits = token_logits + user_logits
            probs, preds = sum_logits.max(dim=-1)
            logits.append(scores)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.end_token_idx).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def decode_beam_search(self, encoder_states, user_embedding, beam=4):
        bsz = encoder_states[0].shape[0]
        xs = self._starts(bsz).reshape(1, bsz, -1)  # (batch_size, _)
        sequences = [[[list(), list(), 1.0]]] * bsz
        for i in range(self.longest_label):
            # at beginning there is 1 candidate, when i!=0 there are 4 candidates
            if i != 0:
                xs = []
                for d in range(len(sequences[0])):
                    for j in range(bsz):
                        text = sequences[j][d][0]
                        xs.append(text)
                xs = torch.stack(xs).reshape(beam, bsz, -1)  # (beam, batch_size, _)

            with torch.no_grad():
                if i == 1:
                    user_embedding = user_embedding.repeat(beam, 1)
                    encoder_states = (encoder_states[0].repeat(beam, 1, 1),
                                      encoder_states[1].repeat(beam, 1, 1))

                scores, _ = self.decoder(xs.reshape(len(sequences[0]) * bsz, -1), encoder_states)
                scores = scores[:, -1:, :]
                token_logits = F.linear(scores, self.token_embedding.weight)
                user_logits = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)
                sum_logits = token_logits + user_logits

            logits = sum_logits.reshape(len(sequences[0]), bsz, 1, -1)
            scores = scores.reshape(len(sequences[0]), bsz, 1, -1)
            logits = torch.nn.functional.softmax(logits)  # turn into probabilities,in case of negative numbers
            probs, preds = logits.topk(beam, dim=-1)
            # (candeidate, bs, 1 , beam) during first loop, candidate=1, otherwise candidate=beam

            for j in range(bsz):
                all_candidates = []
                for n in range(len(sequences[j])):
                    for k in range(beam):
                        prob = sequences[j][n][2]
                        score = sequences[j][n][1]
                        if score == []:
                            score_tmp = scores[n][j][0].unsqueeze(0)
                        else:
                            score_tmp = torch.cat((score, scores[n][j][0].unsqueeze(0)), dim=0)
                        seq_tmp = torch.cat((xs[n][j].reshape(-1), preds[n][j][0][k].reshape(-1)))
                        candidate = [seq_tmp, score_tmp, prob * probs[n][j][0][k]]
                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[2], reverse=True)
                sequences[j] = ordered[:beam]

            # check if everyone has generated an end token
            all_finished = ((xs == self.end_token_idx).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.stack([seq[0][1] for seq in sequences])
        xs = torch.stack([seq[0][0] for seq in sequences])
        return logits, xs

    def converse(self, batch, mode):
        context_tokens, context_entities, response = batch['context_tokens'], batch['context_entities'], batch[
            'response']
        kg_embedding = self.kg_encoder(None, self.edge_idx, self.edge_type)
        user_embedding = self.encode_user(context_entities, kg_embedding)
        encoder_state = self.dialog_encoder(context_tokens)
        if mode != 'test':
            self.longest_label = max(self.longest_label, response.shape[1])
            logits, preds = self.decode_forced(encoder_state, user_embedding, response)
            logits = logits.view(-1, logits.shape[-1])
            labels = response.view(-1)
            return self.conv_loss(logits, labels), preds
        else:
            _, preds = self.decode_greedy(encoder_state, user_embedding)
            return preds

#    def _starts(self, batch_size):
#        """Return bsz start tokens."""
#        return self.START.detach().expand(batch_size, 1)

#    def forward(self, batch, mode, stage):
#        if len(self.gpu) >= 2:
#            self.edge_idx = self.edge_idx.cuda(torch.cuda.current_device())
#            self.edge_type = self.edge_type.cuda(torch.cuda.current_device())
#        if stage == "conv":
#            return self.converse(batch, mode)
#        if stage == "rec":
#            return self.recommend(batch, mode)

#    def freeze_parameters(self):
#        freeze_models = [self.kg_encoder, self.kg_attn, self.rec_bias]
#        for model in freeze_models:
#            for p in model.parameters():
#                p.requires_grad = False