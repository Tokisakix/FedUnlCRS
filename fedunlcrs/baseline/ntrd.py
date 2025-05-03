import os
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch_geometric.nn import GCNConv, RGCNConv

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

class NTRDModel(torch.nn.Module):
    def __init__(self, n_item:int, n_entity:int, n_word:int, model_config:Dict, device:str):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        super(NTRDModel, self).__init__()
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
        self.entity_embedding = nn.Embedding(self.n_entity, self.token_emb_dim, self.pad_token_idx)
        nn.init.normal_(self.entity_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.entity_embedding.weight[self.pad_token_idx], 0)
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
        copy_mask = np.load(os.path.join(self.dpath, "copy_mask.npy")).astype(bool)
        if self.replace_token:
            if self.replace_token_idx < len(copy_mask):
                copy_mask[self.replace_token_idx] = False
            else:
                copy_mask = np.insert(copy_mask,len(copy_mask),False)
        self.copy_mask = torch.as_tensor(copy_mask).to(self.device)
        

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

        logger.debug('[Finish build conv layer]')
    
    def rec_forward(self, batch_data: List[Dict], item_edger: Dict, entity_edger: Dict, word_edger: Dict):
        related_entity = [meta_data["entity"] for meta_data in batch_data]
        related_word = [meta_data["word"] for meta_data in batch_data]
        labels = torch.LongTensor([meta_data["label"] for meta_data in batch_data]).to(self.device)

        max_entity_len = max(len(x) for x in related_entity) if related_entity else 0
        max_word_len = max(len(x) for x in related_word) if related_word else 0

        for i in range(len(related_entity)):
            related_entity[i] += [self.pad_entity_idx] * (max_entity_len - len(related_entity[i]))
        for i in range(len(related_word)):
            related_word[i] += [self.pad_word_idx] * (max_word_len - len(related_word[i]))

        related_entity = torch.LongTensor(related_entity).to(self.device)  # [bs, entity_len]
        related_word = torch.LongTensor(related_word).to(self.device)      # [bs, word_len]

        # entity_padding_mask = related_entity.eq(self.pad_entity_idx)  # [bs, entity_len]
        # word_padding_mask = related_word.eq(self.pad_word_idx)        # [bs, word_len]

        user_embedding = []
        for entity_list, word_list in zip(related_entity, related_word):
            entity_repr = self.entity_embedding.weight[entity_list]
            entity_representations = entity_repr.mean(dim=0, keepdim=True)

            word_repr = self.word_kg_embedding.weight[word_list]
            word_representations = word_repr.mean(dim=0, keepdim=True)

            # entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
            # word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)

            user_rep = self.gate_layer(entity_representations, word_representations)
            user_embedding.append(user_rep)
        user_embedding = torch.concatenate(user_embedding, dim=0)
        rec_scores = F.linear(user_embedding, self.entity_embedding.weight, self.rec_bias.bias)  # (bs, #entity)

        rec_loss = self.rec_loss(rec_scores, labels)

        info_loss_mask = torch.sum(labels)
        if info_loss_mask.item() == 0:
            info_loss = None
        elif False:
            word_info_rep = self.infomax_norm(word_attn_rep)  
            info_predict = F.linear(word_info_rep, entity_repr, self.infomax_bias.bias)  # (bs, #entity)
            labels_one_hot = F.one_hot(labels, num_classes=info_predict.size(1)).float() 
            info_loss = self.infomax_loss(info_predict, labels_one_hot) / info_loss_mask

        return rec_scores, 0.0, rec_loss

    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)
    
    def converse(self, batch, mode):
        context_tokens, context_entities, context_words, response, all_movies = batch

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
            logits, preds,latent = self._decode_forced_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                        entity_padding_mask,
                                                        conv_word_reps, conv_word_emb, word_padding_mask,
                                                        response)

            logits_ = logits.view(-1, logits.shape[-1])
            response_ = response.view(-1)
            gen_loss = self.conv_loss(logits_, response_)

            assert torch.sum(all_movies!=0, dim=(0,1)) == torch.sum((response == 30000), dim=(0,1)) #30000 means the idx of [ITEM]
            masked_for_selection_token = (response == self.replace_token_idx) 

            matching_tensor,_ = self.movie_selector(latent,tokens_encoding,conv_word_reps,word_padding_mask)
            matching_logits_ = self.matching_linear(matching_tensor)

            matching_logits = torch.masked_select(matching_logits_, masked_for_selection_token.unsqueeze(-1).expand_as(matching_logits_)).view(-1, matching_logits_.shape[-1])

            all_movies = torch.masked_select(all_movies,(all_movies != 0)) 
            matching_logits = matching_logits.view(-1,matching_logits.shape[-1])
            all_movies = all_movies.view(-1)
            selection_loss = self.sel_loss(matching_logits,all_movies)
            return gen_loss,selection_loss, preds
        else:
            logits, preds,latent = self._decode_greedy_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                        entity_padding_mask,
                                                        conv_word_reps, conv_word_emb, word_padding_mask)
            
            preds_for_selection = preds[:, 1:] # skip the start_ind
            masked_for_selection_token = (preds_for_selection == self.replace_token_idx)

            matching_tensor,_ = self.movie_selector(latent,tokens_encoding,conv_word_reps,word_padding_mask)
            matching_logits_ = self.matching_linear(matching_tensor)
            matching_logits = torch.masked_select(matching_logits_, masked_for_selection_token.unsqueeze(-1).expand_as(matching_logits_)).view(-1, matching_logits_.shape[-1])

            if matching_logits.shape[0] is not 0:
                    #W1: greedy
                    _, matching_pred = matching_logits.max(dim=-1) # [bsz * dynamic_movie_nums] 
            else:
                matching_pred = None
            return preds,matching_pred,matching_logits_
    
    def _decode_greedy_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                               word_reps, word_emb_attn, word_mask):
        batch_size = token_encoding[0].shape[0]
        inputs = self._starts(batch_size).long()
        incr_state = None
        logits = []
        latents = []
        for _ in range(self.response_truncate):
            dialog_latent, incr_state = self.conv_decoder(inputs, token_encoding, word_reps, word_mask,
                                                          entity_reps, entity_mask, incr_state)
            dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim)
            latents.append(dialog_latent)
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
        latents = torch.cat(latents, dim=1)
        return logits, inputs, latents

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
        return sum_logits, preds, dialog_latent