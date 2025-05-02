import torch
from loguru import logger
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from typing import Dict, List


class GRU4RECModel(torch.nn.Module):
    """

    Attributes:
        item_size: A integer indicating the number of items.
        hidden_size: A integer indicating the hidden state size in GRU.
        num_layers: A integer indicating the number of GRU layers.
        dropout_hidden: A float indicating the dropout rate to dropout hidden state.
        dropout_input: A integer indicating if we dropout the input of model.
        embedding_dim: A integer indicating the dimension of item embedding.
        batch_size: A integer indicating the batch size.

    """

    def __init__(self, n_item:int, n_entity:int, n_word:int, model_config:Dict, device:str):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        super(GRU4RECModel, self).__init__()
        self.device = device
        self.item_size = n_item
        self.hidden_size = model_config['gru_hidden_size']
        self.num_layers = model_config['num_layers']
        self.dropout_hidden = model_config['dropout_hidden']
        self.dropout_input = model_config['dropout_input']
        self.embedding_dim = model_config['embedding_dim']
        self.batch_size = model_config['batch_size']
        self.max_seq_length = 100
        self.build_model()

    def build_model(self):
        self.item_embeddings = nn.Embedding(self.item_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim,
                          self.hidden_size,
                          self.num_layers,
                          dropout=self.dropout_hidden,
                          batch_first=True)
        self.rec_linear = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.item_size)

        logger.debug('[Finish build rec layer]')

    def reconstruct_input(self, input_ids):
        """
        convert the padding from left to right
        """

        def reverse_padding(ids):
            ans = [0] * len(ids)
            idx = 0
            for m_id in ids:
                m_id = m_id.item()
                if m_id != 0:
                    ans[idx] = m_id
                    idx += 1
            return ans

        input_len = [torch.sum((ids != 0).long()).item() for ids in input_ids]
        input_ids = [reverse_padding(ids) for ids in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = (input_ids != 0).long()

        return input_ids.to(self.device), input_len, input_mask.to(self.device)

    def cross_entropy(self, seq_out, pos_ids, neg_ids, input_mask):
        # [batch seq_len hidden_size]
        pos_emb = self.item_embeddings(pos_ids)
        neg_emb = self.item_embeddings(neg_ids)

        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))

        # [batch*seq_len hidden_size]
        seq_emb = seq_out.contiguous().view(-1, self.hidden_size)

        # [batch*seq_len]
        pos_logits = torch.sum(pos * seq_emb, -1)
        neg_logits = torch.sum(neg * seq_emb, -1)

        # [batch*seq_len]
        istarget = (pos_ids > 0).view(pos_ids.size(0) * pos_ids.size(1)).float()
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def rec_forward(self, batch, item_edger, entity_edger, word_edger):
        """
        Args:
            input_ids: padding in left, [pad, pad, id1, id2, ..., idn]
            target_ids: padding in left, [pad, pad, id2, id3, ..., y]
        """
        input_ids, input_mask, labels = [], [], []

        for meta_data in batch:
            item_seq = meta_data["item"]
            if len(item_seq) > self.max_seq_length:
                item_seq = item_seq[-self.max_seq_length:]
            padding_len = self.max_seq_length - len(item_seq)
            input_id = item_seq + [0] * padding_len
            mask = [1] * len(item_seq) + [0] * padding_len
            label = meta_data["label"]

            input_ids.append(input_id)
            input_mask.append(mask)
            labels.append(label)

        input_ids = torch.LongTensor(input_ids).to(self.device)         # (batch_size, max_seq_len)
        input_mask = torch.LongTensor(input_mask).to(self.device)       # (batch_size, max_seq_len)
        labels = torch.LongTensor(labels).to(self.device)               # (batch_size,)

        context = input_ids.clone()

        input_ids, input_len, input_mask = self.reconstruct_input(input_ids)

        embedded = self.item_embeddings(context)  # (batch, seq_len, hidden_size)
        output, hidden = self.gru(embedded)
        # output, output_len = pad_packed_sequence(output, batch_first=True)

        batch, seq_len, hidden_size = output.size()
        logit = output.view(batch, seq_len, hidden_size)

        last_logit = logit[:, -1, :]
        rec_scores = torch.nn.functional.linear(last_logit, self.item_embeddings.weight, self.rec_linear.bias)
        rec_scores = rec_scores.squeeze(1)

        # max_out_len = max([len_ for len_ in output_len])
        rec_loss = torch.nn.functional.cross_entropy(rec_scores, labels)

        return rec_scores, labels, rec_loss
