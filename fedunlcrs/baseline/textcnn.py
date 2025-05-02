import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from typing import Dict, List

class TextCNNModel(torch.nn.Module):
    """
        
    Attributes:
        movie_num: A integer indicating the number of items.
        num_filters: A string indicating the number of filter in CNN.
        embed: A integer indicating the size of embedding layer.
        filter_sizes: A string indicating the size of filter in CNN.
        dropout: A float indicating the dropout rate.

    """

    def __init__(self, n_item:int, n_entity:int, n_word:int, model_config:Dict, device:str):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        super(TextCNNModel, self).__init__()
        self.device = device
        self.movie_num = n_item
        self.num_filters = model_config['num_filters']
        self.embed = model_config['embed']
        self.filter_sizes = eval(model_config['filter_sizes'])
        self.dropout = 0.2
        self.max_seq_length = model_config['max_history_items']
        self.build_model()

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

        # this loss may conduct to some weakness
        self.rec_loss = nn.CrossEntropyLoss()

        logger.debug('[Finish build rec layer]')

    def rec_forward(self, batch_data: List[Dict], item_edger: Dict, entity_edger: Dict, word_edger: Dict):
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

        input_ids = torch.LongTensor(input_ids).to(self.device)         # (batch_size, max_seq_len)
        input_mask = torch.LongTensor(input_mask).to(self.device)       # (batch_size, max_seq_len)
        labels = torch.LongTensor(labels).to(self.device)               # (batch_size,)

        context = input_ids.clone()

        out = self.embedding(context)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)

        rec_scores = out
        rec_loss = self.rec_loss(out, labels)

        return rec_scores, rec_scores, rec_loss