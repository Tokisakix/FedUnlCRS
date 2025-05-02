import torch
from loguru import logger
from torch import nn
from .rec import SASREC
from typing import Dict, List

class SASRECModel(torch.nn.Module):
    """
        
    Attributes:
        hidden_dropout_prob: A float indicating the dropout rate to dropout hidden state in SASRec.
        initializer_range: A float indicating the range of parameters initiation in SASRec.
        hidden_size: A integer indicating the size of hidden state in SASRec.
        max_seq_length: A integer indicating the max interaction history length.
        item_size: A integer indicating the number of items.
        num_attention_heads: A integer indicating the head number in SASRec.
        attention_probs_dropout_prob: A float indicating the dropout rate in attention layers.
        hidden_act: A string indicating the activation function type in SASRec.
        num_hidden_layers: A integer indicating the number of hidden layers in SASRec.

    """

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
        self.build_model()

    def build_model(self):
        # build BERT layer, give the architecture, load pretrained parameters
        self.SASREC = SASREC(self.hidden_dropout_prob, self.device,
                             self.initializer_range, self.hidden_size,
                             self.max_seq_length, self.item_size,
                             self.num_attention_heads,
                             self.attention_probs_dropout_prob,
                             self.hidden_act, self.num_hidden_layers)
        self.mlp = nn.Linear(self.hidden_size, self.item_size)

        # this loss may conduct to some weakness
        self.rec_loss = nn.CrossEntropyLoss()

        logger.debug('[Finish build rec layer]')
    
    def rec_forward(self, batch_data:List[Dict], item_edger:Dict, entity_edger:Dict, word_edger:Dict):
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