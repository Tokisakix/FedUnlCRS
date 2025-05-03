import torch
from loguru import logger
from torch import nn
from .rec import SASREC
from typing import Dict, List, Tuple

class SASRECModel(torch.nn.Module):

    def __init__(self, n_item:int, n_entity:int, n_word:int, model_config:Dict, device:str):
        super(SASRECModel, self).__init__()
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
        self.n_word = n_word
        self.build_model()

    def build_model(self):
        # build BERT layer, give the architecture, load pretrained parameters
        self.SASREC = SASREC(self.hidden_dropout_prob, self.device,
                             self.initializer_range, self.hidden_size,
                             self.max_seq_length, self.item_size,
                             self.num_attention_heads,
                             self.attention_probs_dropout_prob,
                             self.hidden_act, self.num_hidden_layers)
        self.con_SASREC = SASREC(self.hidden_dropout_prob, self.device,
                             self.initializer_range, self.hidden_size,
                             self.max_seq_length, self.n_word,
                             self.num_attention_heads,
                             self.attention_probs_dropout_prob,
                             self.hidden_act, self.num_hidden_layers)
        self.mlp = nn.Linear(self.hidden_size, self.item_size)
        self.con = nn.Linear(self.hidden_size, self.n_word)

        # this loss may conduct to some weakness
        self.rec_loss = nn.CrossEntropyLoss()

        logger.debug('[Finish build rec layer]')
    
    def rec_forward(self, batch_data:List[Dict], 
                    item_edger:Dict, entity_edger:Dict, word_edger:Dict):
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
        # bert_embed = self.bert(context, attention_mask=mask).pooler_output
        sequence_output = self.SASREC(input_ids, input_mask)
        sas_embed = sequence_output[:, -1, :]

        # embed = torch.cat((sas_embed, bert_embed), dim=1)
        rec_scores = self.mlp(sas_embed)

        rec_loss = self.rec_loss(rec_scores, labels)

        return rec_scores, rec_scores, rec_loss
    
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
            
            #last_hidden_state = self.bert(input_ids, attention_mask=input_mask).last_hidden_state[0]
            con_sequence_output = self.con_SASREC(input_ids, input_mask)
            logit = self.con(con_sequence_output)[0, :label.size(0)]
            logits.append(logit)

        labels = torch.concatenate(labels, dim=0)
        logits = torch.concatenate(logits, dim=0)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        response = torch.argmax(logits, dim=-1)

        return response, labels, loss