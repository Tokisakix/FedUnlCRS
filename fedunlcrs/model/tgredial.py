from typing import Dict, List, Tuple
import torch
from torch import nn
from transformers import BertConfig, BertModel
from .rec import SASREC, ConversationModule

dataset_language_map = {
    'ReDial': 'en',
    'TGReDial': 'zh',
    'GoRecDial': 'en',
    'OpenDialKG': 'en',
    'Inspired': 'en',
    'DuRecDial': 'zh'
}

class TGRecModel(torch.nn.Module):
    def __init__(
            self, n_item:int, n_entity:int, n_word:int,
            model_config:Dict, device:str
        ) -> None:
        super(TGRecModel, self).__init__()
        self.n_word = n_word
        self.hidden_dropout_prob = model_config['hidden_dropout_prob']
        self.initializer_range = model_config['initializer_range']
        self.hidden_size = model_config['hidden_size']
        self.max_seq_length = model_config['max_history_items']
        self.item_size = n_entity + 1
        self.num_attention_heads = model_config['num_attention_heads']
        self.attention_probs_dropout_prob = model_config['attention_probs_dropout_prob']
        self.hidden_act = model_config['hidden_act']
        self.num_hidden_layers = model_config['num_hidden_layers']
        self.device = device
        self.build_model()
        return

    def build_model(self):
        config = BertConfig(vocab_size=self.item_size) 
        self.bert = BertModel(config)  
        self.bert_hidden_size = self.bert.config.hidden_size
        self.concat_embed_size = self.bert_hidden_size + self.hidden_size
        self.user_emb_dim = self.bert_hidden_size + self.hidden_size
        self.fusion = nn.Linear(self.concat_embed_size, self.item_size)
        self.SASREC = SASREC(self.hidden_dropout_prob, self.device,
                             self.initializer_range, self.hidden_size,
                             self.max_seq_length, self.item_size,
                             self.num_attention_heads,
                             self.attention_probs_dropout_prob,
                             self.hidden_act, self.num_hidden_layers)

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
        mask = input_mask.clone()
        bert_embed = self.bert(context, attention_mask=mask).pooler_output
        sequence_output = self.SASREC(input_ids, input_mask)
        sas_embed = sequence_output[:, -1, :]

        embed = torch.cat((sas_embed, bert_embed), dim=1)
        rec_scores = self.fusion(embed)

        rec_loss = self.rec_loss(rec_scores, labels)

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
        mask = input_mask.clone()
        bert_embed = self.bert(context, attention_mask=mask).pooler_output
        sequence_output = self.SASREC(input_ids, input_mask)
        sas_embed = sequence_output[:, -1, :]

        user_embeddings = torch.cat((sas_embed, bert_embed), dim=1)

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