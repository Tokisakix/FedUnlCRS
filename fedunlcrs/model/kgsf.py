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

    def _build_conversation_layer(self):
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))
        self.conv_encoder = TransformerEncoder(
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            embedding_size=self.token_emb_dim,
            ffn_size=self.ffn_size,
            vocabulary_size=self.vocab_size,
            embedding=self.token_embedding,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            padding_idx=self.pad_token_idx,
            learn_positional_embeddings=self.learn_positional_embeddings,
            embeddings_scale=self.embeddings_scale,
            reduction=self.reduction,
            n_positions=self.n_positions,
        )

        self.conv_entity_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_entity_attn_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_word_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_word_attn_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)

        self.copy_norm = nn.Linear(self.ffn_size * 3, self.token_emb_dim)
        self.copy_output = nn.Linear(self.token_emb_dim, self.vocab_size)
        self.copy_mask = torch.as_tensor(np.load(os.path.join(self.dpath, "copy_mask.npy")).astype(bool),
                                         ).to(self.device)

        self.conv_decoder = TransformerDecoderKG(
            self.n_heads, self.n_layers, self.token_emb_dim, self.ffn_size, self.vocab_size,
            embedding=self.token_embedding,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            embeddings_scale=self.embeddings_scale,
            learn_positional_embeddings=self.learn_positional_embeddings,
            padding_idx=self.pad_token_idx,
            n_positions=self.n_positions
        )
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)

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

    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)

    def _decode_forced_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                               word_reps, word_emb_attn, word_mask, response):
        batch_size, seq_len = response.shape
        start = self._starts(batch_size)
        inputs = torch.cat((start, response[:, :-1]), dim=-1).long()

        dialog_latent, _ = self.conv_decoder(inputs, token_encoding, word_reps, word_mask,
                                             entity_reps, entity_mask)  # (bs, seq_len, dim)
        entity_latent = entity_emb_attn.unsqueeze(1).expand(-1, seq_len, -1)
        word_latent = word_emb_attn.unsqueeze(1).expand(-1, seq_len, -1)
        copy_latent = self.copy_norm(
            torch.cat((entity_latent, word_latent, dialog_latent), dim=-1))  # (bs, seq_len, dim)

        copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(
            0)  # (bs, seq_len, vocab_size)
        gen_logits = F.linear(dialog_latent, self.token_embedding.weight)  # (bs, seq_len, vocab_size)
        sum_logits = copy_logits + gen_logits
        preds = sum_logits.argmax(dim=-1)
        return sum_logits, preds

    def _decode_greedy_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                               word_reps, word_emb_attn, word_mask):
        batch_size = token_encoding[0].shape[0]
        inputs = self._starts(batch_size).long()
        incr_state = None
        logits = []
        for _ in range(self.response_truncate):
            dialog_latent, incr_state = self.conv_decoder(inputs, token_encoding, word_reps, word_mask,
                                                          entity_reps, entity_mask, incr_state)
            dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim)
            db_latent = entity_emb_attn.unsqueeze(1)
            concept_latent = word_emb_attn.unsqueeze(1)
            copy_latent = self.copy_norm(torch.cat((db_latent, concept_latent, dialog_latent), dim=-1))

            copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(0)
            gen_logits = F.linear(dialog_latent, self.token_embedding.weight)
            sum_logits = copy_logits + gen_logits
            preds = sum_logits.argmax(dim=-1).long()
            logits.append(sum_logits)
            inputs = torch.cat((inputs, preds), dim=1)

            finished = ((inputs == self.end_token_idx).sum(dim=-1) > 0).sum().item() == batch_size
            if finished:
                break
        logits = torch.cat(logits, dim=1)
        return logits, inputs

    def _decode_beam_search_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                                    word_reps, word_emb_attn, word_mask, beam=4):
        batch_size = token_encoding[0].shape[0]
        inputs = self._starts(batch_size).long().reshape(1, batch_size, -1)
        incr_state = None

        sequences = [[[list(), list(), 1.0]]] * batch_size
        for i in range(self.response_truncate):
            if i == 1:
                token_encoding = (token_encoding[0].repeat(beam, 1, 1),
                                  token_encoding[1].repeat(beam, 1, 1))
                entity_reps = entity_reps.repeat(beam, 1, 1)
                entity_emb_attn = entity_emb_attn.repeat(beam, 1)
                entity_mask = entity_mask.repeat(beam, 1)
                word_reps = word_reps.repeat(beam, 1, 1)
                word_emb_attn = word_emb_attn.repeat(beam, 1)
                word_mask = word_mask.repeat(beam, 1)

            # at beginning there is 1 candidate, when i!=0 there are 4 candidates
            if i != 0:
                inputs = []
                for d in range(len(sequences[0])):
                    for j in range(batch_size):
                        text = sequences[j][d][0]
                        inputs.append(text)
                inputs = torch.stack(inputs).reshape(beam, batch_size, -1)  # (beam, batch_size, _)

            with torch.no_grad():
                dialog_latent, incr_state = self.conv_decoder(
                    inputs.reshape(len(sequences[0]) * batch_size, -1),
                    token_encoding, word_reps, word_mask,
                    entity_reps, entity_mask, incr_state
                )
                dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim)
                db_latent = entity_emb_attn.unsqueeze(1)
                concept_latent = word_emb_attn.unsqueeze(1)
                copy_latent = self.copy_norm(torch.cat((db_latent, concept_latent, dialog_latent), dim=-1))

                copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(0)
                gen_logits = F.linear(dialog_latent, self.token_embedding.weight)
                sum_logits = copy_logits + gen_logits

            logits = sum_logits.reshape(len(sequences[0]), batch_size, 1, -1)
            # turn into probabilities,in case of negative numbers
            probs, preds = torch.nn.functional.softmax(logits).topk(beam, dim=-1)

            # (candeidate, bs, 1 , beam) during first loop, candidate=1, otherwise candidate=beam

            for j in range(batch_size):
                all_candidates = []
                for n in range(len(sequences[j])):
                    for k in range(beam):
                        prob = sequences[j][n][2]
                        logit = sequences[j][n][1]
                        if logit == []:
                            logit_tmp = logits[n][j][0].unsqueeze(0)
                        else:
                            logit_tmp = torch.cat((logit, logits[n][j][0].unsqueeze(0)), dim=0)
                        seq_tmp = torch.cat((inputs[n][j].reshape(-1), preds[n][j][0][k].reshape(-1)))
                        candidate = [seq_tmp, logit_tmp, prob * probs[n][j][0][k]]
                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[2], reverse=True)
                sequences[j] = ordered[:beam]

            # check if everyone has generated an end token
            all_finished = ((inputs == self.end_token_idx).sum(dim=1) > 0).sum().item() == batch_size
            if all_finished:
                break
        logits = torch.stack([seq[0][1] for seq in sequences])
        inputs = torch.stack([seq[0][0] for seq in sequences])
        return logits, inputs

    def converse(self, batch, mode):
        context_tokens, context_entities, context_words, response = batch

        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        entity_padding_mask = context_entities.eq(self.pad_entity_idx)  # (bs, entity_len)
        word_padding_mask = context_words.eq(self.pad_word_idx)  # (bs, seq_len)

        entity_representations = entity_graph_representations[context_entities]
        word_representations = word_graph_representations[context_words]

        entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)

        # encoder-decoder
        tokens_encoding = self.conv_encoder(context_tokens)
        conv_entity_emb = self.conv_entity_attn_norm(entity_attn_rep)
        conv_word_emb = self.conv_word_attn_norm(word_attn_rep)
        conv_entity_reps = self.conv_entity_norm(entity_representations)
        conv_word_reps = self.conv_word_norm(word_representations)
        if mode != 'test':
            logits, preds = self._decode_forced_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                        entity_padding_mask,
                                                        conv_word_reps, conv_word_emb, word_padding_mask,
                                                        response)

            logits = logits.view(-1, logits.shape[-1])
            response = response.view(-1)
            loss = self.conv_loss(logits, response)
            return loss, preds
        else:
            logits, preds = self._decode_greedy_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                        entity_padding_mask,
                                                        conv_word_reps, conv_word_emb, word_padding_mask)
            return preds
