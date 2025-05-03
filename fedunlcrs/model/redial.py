import torch.nn as nn
from typing import Dict, List
import torch
import torch.nn.functional as F

class ReDialRecModel(torch.nn.Module):
    """

    Attributes:
        n_entity: A integer indicating the number of entities.
        layer_sizes: A integer indicating the size of layer in autorec.
        pad_entity_idx: A integer indicating the id of entity padding.

    """

    def __init__(self, n_item:int, n_entity:int, n_word:int, model_config:Dict, device:str):
        super(ReDialRecModel, self).__init__()
        self.device = device
        self.n_entity = n_entity
        self.layer_sizes = model_config['autorec_layer_sizes']
        self.pad_entity_idx = 0
        self.autorec_f = model_config['autorec_f']
        self.autorec_g = model_config['autorec_g']
        self.emb_dim = model_config["emb_dim"]
        self.n_item = n_item
        self.context_truncate = opt['context_truncate']
        self.response_truncate = opt['response_truncate']
        self.pad_id = vocab['pad']
        
        self.utterance_encoder_hidden_size = opt['utterance_encoder_hidden_size']
        self.dialog_encoder_hidden_size = opt['dialog_encoder_hidden_size']
        self.dialog_encoder_num_layers = opt['dialog_encoder_num_layers']
        self.use_dropout = opt['use_dropout']
        self.dropout = opt['dropout']
        # SwitchingDecoder
        self.decoder_hidden_size = opt['decoder_hidden_size']
        self.decoder_num_layers = opt['decoder_num_layers']
        self.decoder_embedding_dim = opt['decoder_embedding_dim']
        self.build_model()


    def build_model(self):
        # AutoRec
        if self.autorec_f == 'identity':
            self.f = lambda x: x
        elif self.autorec_f == 'sigmoid':
            self.f = nn.Sigmoid()
        elif self.autorec_f == 'relu':
            self.f = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for f : {}".format(self.opt['autorec_f']))

        if self.autorec_g == 'identity':
            self.g = lambda x: x
        elif self.autorec_g == 'sigmoid':
            self.g = nn.Sigmoid()
        elif self.autorec_g == 'relu':
            self.g = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for g : {}".format(self.opt['autorec_g']))

        self.encoder = nn.ModuleList([nn.Linear(self.n_entity, self.layer_sizes[0]) if i == 0
                                      else nn.Linear(self.layer_sizes[i - 1], self.layer_sizes[i])
                                      for i in range(len(self.layer_sizes))])
        self.user_repr_dim = self.layer_sizes[-1]
        self.decoder = nn.Linear(self.user_repr_dim, self.n_entity)
        self.loss = nn.CrossEntropyLoss()
        self.rec_bias = torch.nn.Linear(self.emb_dim, self.n_entity)
        self.entity_embedding = torch.nn.Embedding(self.n_item, self.emb_dim, 0)
        self.item_embedding = torch.nn.Embedding(self.n_item, self.emb_dim, 0)

        # add
        self.encoder = HRNN(
            embedding=embedding,
            utterance_encoder_hidden_size=self.utterance_encoder_hidden_size,
            dialog_encoder_hidden_size=self.dialog_encoder_hidden_size,
            dialog_encoder_num_layers=self.dialog_encoder_num_layers,
            use_dropout=self.use_dropout,
            dropout=self.dropout,
            pad_token_idx=self.pad_token_idx
        )

        self.decoder = SwitchingDecoder(
            hidden_size=self.decoder_hidden_size,
            context_size=self.dialog_encoder_hidden_size,
            num_layers=self.decoder_num_layers,
            vocab_size=self.vocab_size,
            embedding=embedding,
            pad_token_idx=self.pad_token_idx
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)

    def rec_forward(self, batch_data:List[Dict], item_edger:Dict, entity_edger:Dict, word_edger:Dict):
        """

        Args:
            batch: ::

                {
                    'context_entities': (batch_size, n_entity),
                    'item': (batch_size)
                }

            mode (str)

        """
        related_entity = [meta_data["entity"] for meta_data in batch_data]
        related_item = [meta_data["item"] for meta_data in batch_data]
        labels = torch.LongTensor([meta_data["label"] for meta_data in batch_data]).to(self.device)

        entity_embedding = []
        item_embedding = []
        for entity_list, item_list in zip(related_entity, related_item):
            entity_tensor = torch.LongTensor(entity_list).to(self.device)
            entity_repr = self.entity_embedding(entity_tensor).mean(dim=0, keepdim=True)
            entity_embedding.append(entity_repr)

            item_tensor = torch.LongTensor(item_list).to(self.device)
            item_repr = self.item_embedding(item_tensor).mean(dim=0, keepdim=True)
            item_embedding.append(item_repr)

        entity_embedding = torch.concatenate(entity_embedding, dim=0)
        item_embedding = torch.concatenate(item_embedding, dim=0)
        combined_embedding = (entity_embedding + item_embedding) / 2
        logits = self.decoder(combined_embedding)
        loss = self.loss(logits, labels)
        return logits, labels, loss

    
    #add
    def forward(self, batch, mode):
        """
        Args:
            batch: ::

                {
                    'context': (batch_size, max_context_length, max_utterance_length),
                    'context_lengths': (batch_size),
                    'utterance_lengths': (batch_size, max_context_length),
                    'request': (batch_size, max_utterance_length),
                    'request_lengths': (batch_size),
                    'response': (batch_size, max_utterance_length)
                }

        """
        assert mode in ('train', 'valid', 'test')
        if mode == 'train':
            self.train()
        else:
            self.eval()

        context = batch['context']
        utterance_lengths = batch['utterance_lengths']
        context_lengths = batch['context_lengths']
        context_state = self.encoder(context, utterance_lengths,
                                     context_lengths)  # (batch_size, context_encoder_hidden_size)

        request = batch['request']
        request_lengths = batch['request_lengths']
        log_probs = self.decoder(request, request_lengths,
                                 context_state)  # (batch_size, max_utterance_length, vocab_size + 1)
        preds = log_probs.argmax(dim=-1)  # (batch_size, max_utterance_length)

        log_probs = log_probs.view(-1, log_probs.shape[-1])
        response = batch['response'].view(-1)
        loss = self.loss(log_probs, response)

        return loss, preds