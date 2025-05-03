import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Tuple

from .rec import ConversationModule

class TextCNNModel(torch.nn.Module):
    def __init__(
            self, n_item:int, n_entity:int, n_word:int,
            model_config:Dict, device:str
        ) -> None:
        super(TextCNNModel, self).__init__()
        self.n_word = n_word
        self.device = device
        self.movie_num = n_item
        self.num_filters = model_config['num_filters']
        self.embed = model_config['embed']
        self.filter_sizes = eval(model_config['filter_sizes'])
        self.dropout = 0.2
        self.max_seq_length = model_config['max_history_items']
        self.user_emb_dim = self.num_filters * len(self.filter_sizes)
        self.build_model()
        return

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def build_model(self):
        self.embedding = nn.Embedding(self.movie_num, self.embed)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embed)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.movie_num)

        self.rec_loss = nn.CrossEntropyLoss()
        self.con_module = ConversationModule(self.user_emb_dim, self.n_word, self.device)
        return

    def rec_forward(
            self, batch_data:List[Dict],
            item_edger:Dict, entity_edger:Dict, word_edger:Dict
        ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        input_ids, input_mask, labels = [], [], []

        for meta_data in batch_data:
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

        input_ids = torch.LongTensor(input_ids).to(self.device)
        input_mask = torch.LongTensor(input_mask).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)

        context = input_ids.clone()

        out = self.embedding(context)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)

        rec_scores = out
        rec_loss = self.rec_loss(out, labels)

        return rec_scores, rec_scores, rec_loss

    def con_forward(
            self, batch_data:List[Dict],
            item_edger:Dict, entity_edger:Dict, word_edger:Dict
        ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        input_ids, input_mask, labels = [], [], []
        related_words = [meta_data["word"] for meta_data in batch_data]
        texts = [meta_data["text"][1:] for meta_data in batch_data]

        for meta_data in batch_data:
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

        input_ids = torch.LongTensor(input_ids).to(self.device)
        input_mask = torch.LongTensor(input_mask).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)

        context = input_ids.clone()

        out = self.embedding(context)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        user_embeddings = self.dropout(out)

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