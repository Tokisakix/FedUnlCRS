import os
from typing import Dict, List
import torch
from loguru import logger
from torch import nn
from transformers import BertConfig, BertModel
from .rec import SASREC

dataset_language_map = {
    'ReDial': 'en',
    'TGReDial': 'zh',
    'GoRecDial': 'en',
    'OpenDialKG': 'en',
    'Inspired': 'en',
    'DuRecDial': 'zh'
}


class TGRecModel(torch.nn.Module):
    def __init__(self, n_item:int, n_entity:int, n_word:int, model_config:Dict, device:str):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        super(TGRecModel, self).__init__()
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
        # build BERT layer, give the architecture, load pretrained parameters
        config = BertConfig(vocab_size=self.item_size) 
        self.bert = BertModel(config)  
        self.bert_hidden_size = self.bert.config.hidden_size
        self.concat_embed_size = self.bert_hidden_size + self.hidden_size
        self.fusion = nn.Linear(self.concat_embed_size, self.item_size)
        self.SASREC = SASREC(self.hidden_dropout_prob, self.device,
                             self.initializer_range, self.hidden_size,
                             self.max_seq_length, self.item_size,
                             self.num_attention_heads,
                             self.attention_probs_dropout_prob,
                             self.hidden_act, self.num_hidden_layers)

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
        bert_embed = self.bert(context, attention_mask=mask).pooler_output
        sequence_output = self.SASREC(input_ids, input_mask)
        sas_embed = sequence_output[:, -1, :]

        embed = torch.cat((sas_embed, bert_embed), dim=1)
        rec_scores = self.fusion(embed)

        rec_loss = self.rec_loss(rec_scores, labels)

        return rec_scores, rec_scores, rec_loss