import torch
from torch import nn
from typing import Dict, List, Tuple

class GRU4RECModel(torch.nn.Module):
    def __init__(
            self, n_item:int, n_entity:int, n_word:int,
            model_config:Dict, device:str
        ) -> None:
        super(GRU4RECModel, self).__init__()
        self.device = device
        self.n_item = n_item
        self.n_word = n_word
        self.hidden_size = model_config['gru_hidden_size']
        self.num_layers = model_config['num_layers']
        self.dropout_hidden = model_config['dropout_hidden']
        self.dropout_input = model_config['dropout_input']
        self.embedding_dim = model_config['embedding_dim']
        self.batch_size = model_config['batch_size']
        self.max_seq_length = 100
        self.build_model()
        return

    def build_model(self):
        self.text_embeddings = nn.Embedding(self.n_word, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim,
                          self.hidden_size,
                          self.num_layers,
                          dropout=self.dropout_hidden,
                          batch_first=True)
        self.rec_linear = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.n_item)
        self.con_linear = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.n_word)
        return

    def rec_forward(
            self, batch_data:List[Dict],
            item_edger:Dict, entity_edger:Dict, word_edger:Dict
        ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        input_ids, input_mask, labels = [], [], []

        for meta_data in batch_data:
            item_seq = meta_data["text"]
            if len(item_seq) > self.max_seq_length:
                item_seq = item_seq[-self.max_seq_length:]
            padding_len = self.max_seq_length - len(item_seq)
            input_id = item_seq + [0] * padding_len
            mask = [1] * len(item_seq) + [0] * padding_len
            label = meta_data["label"]

            input_ids.append(input_id)
            input_mask.append(mask)
            labels.append(label)

        input_ids = torch.LongTensor(input_ids).to(self.device)
        input_mask = torch.LongTensor(input_mask).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)

        context = input_ids.clone()

        embedded = self.text_embeddings(context)
        output, hidden = self.gru(embedded)

        batch, seq_len, hidden_size = output.size()
        logit = output.view(batch, seq_len, hidden_size)

        last_logit = logit[:, -1, :]
        rec_scores = self.rec_linear(last_logit)
        rec_scores = rec_scores.squeeze(1)

        rec_loss = torch.nn.functional.cross_entropy(rec_scores, labels)

        return rec_scores, labels, rec_loss

    def con_forward(
            self, batch_data:List[Dict],
            item_edger:Dict, entity_edger:Dict, word_edger:Dict
        ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        labels = []
        logits = []

        for meta_data in batch_data:
            item_seq = meta_data["text"]

            if len(item_seq) > self.max_seq_length:
                item_seq = item_seq[-self.max_seq_length:]
            padding_len = self.max_seq_length - len(item_seq)

            input_ids = item_seq + [0] * padding_len
            input_mask = [1] * len(item_seq) + [0] * padding_len
            input_ids = torch.LongTensor([input_ids]).to(self.device)
            input_mask = torch.LongTensor([input_mask]).to(self.device)

            label = torch.LongTensor(meta_data["text"][1:]).to(self.device)
            labels.append(label)
            
            embedded = self.text_embeddings(input_ids)
            output, hidden = self.gru(embedded)

            batch, seq_len, hidden_size = output.size()
            logit = output.view(seq_len, hidden_size)[:label.size(0)]
            logit = self.con_linear(logit)

            logits.append(logit)

        labels = torch.concatenate(labels, dim=0)
        logits = torch.concatenate(logits, dim=0)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        response = torch.argmax(logits, dim=-1)

        return response, labels, loss