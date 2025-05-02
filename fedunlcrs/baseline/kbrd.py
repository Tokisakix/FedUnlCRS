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