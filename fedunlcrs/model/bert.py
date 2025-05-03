import torch
from torch import nn
from transformers import BertConfig, BertModel
from typing import Dict, List, Tuple

class BERTModel(torch.nn.Module):
    def __init__(
            self, n_item:int, n_entity:int, n_word:int,
            model_config:Dict, device:str
        ) -> None:
        super(BERTModel, self).__init__()
        self.n_item = n_item
        self.n_word = n_word
        self.max_seq_length = model_config["max_history_items"]
        self.device = device
        self.build_model()
        return

    def build_model(self):
        config = BertConfig(
            vocab_size=self.n_word,
        )
        self.bert = BertModel(config)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.mlp = nn.Linear(self.bert_hidden_size, self.n_item)
        self.con = nn.Linear(self.bert_hidden_size, self.n_word)

        self.rec_loss = nn.CrossEntropyLoss()
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
        mask = input_mask.clone()
        bert_embed = self.bert(context, attention_mask=mask).pooler_output
        embed = bert_embed
        rec_scores = self.mlp(embed)

        rec_loss = self.rec_loss(rec_scores, labels)

        return rec_scores, None, rec_loss
    
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
            
            last_hidden_state = self.bert(input_ids, attention_mask=input_mask).last_hidden_state[0]
            bert_edmbedding = last_hidden_state[:label.size(0)]
            logit = self.con(bert_edmbedding)
            logits.append(logit)

        labels = torch.concatenate(labels, dim=0)
        logits = torch.concatenate(logits, dim=0)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        response = torch.argmax(logits, dim=-1)

        return response, labels, loss