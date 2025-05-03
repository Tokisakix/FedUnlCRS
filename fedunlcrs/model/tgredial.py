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
        self.context_truncate = opt['context_truncate']
        self.response_truncate = opt['response_truncate']
        self.pad_id = vocab['pad']
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

    def forward(self, batch, mode):
        if mode == 'test' or mode == 'infer':
            enhanced_context = batch[1]
            return self.generate(enhanced_context)
        else:
            enhanced_input_ids = batch[0]
            # torch.tensor's shape = (bs, seq_len, v_s); tuple's length = 12
            lm_logits = self.model(enhanced_input_ids).logits

            # index from 1 to self.reponse_truncate is valid response
            loss = self.calculate_loss(
                lm_logits[:, -self.response_truncate:-1, :],
                enhanced_input_ids[:, -self.response_truncate + 1:])

            pred = torch.max(lm_logits, dim=2)[1]  # [bs, seq_len]
            pred = pred[:, -self.response_truncate:]

            return loss, pred

    def generate(self, context):
        """
        Args:
            context: torch.tensor, shape=(bs, context_turncate)

        Returns:
            generated_response: torch.tensor, shape=(bs, reponse_turncate-1)
        """
        generated_response = []
        former_hidden_state = None
        context = context[..., -self.response_truncate + 1:]

        for i in range(self.response_truncate - 1):
            outputs = self.model(context, former_hidden_state)  # (bs, c_t, v_s),
            last_hidden_state, former_hidden_state = outputs.logits, outputs.past_key_values

            next_token_logits = last_hidden_state[:, -1, :]  # (bs, v_s)
            preds = next_token_logits.argmax(dim=-1).long()  # (bs)

            context = preds.unsqueeze(1)
            generated_response.append(preds)

        generated_response = torch.stack(generated_response).T

        return generated_response

    def generate_bs(self, context, beam=4):
        context = context[..., -self.response_truncate + 1:]
        context_former = context
        batch_size = context.shape[0]
        sequences = [[[list(), 1.0]]] * batch_size
        for i in range(self.response_truncate - 1):
            if sequences != [[[list(), 1.0]]] * batch_size:
                context = []
                for i in range(batch_size):
                    for cand in sequences[i]:
                        text = torch.cat(
                            (context_former[i], torch.tensor(cand[0]).to(self.device)))  # 由于取消了state，与之前的context拼接
                        context.append(text)
                context = torch.stack(context)
            with torch.no_grad():
                outputs = self.model(context)
            last_hidden_state, state = outputs.logits, outputs.past_key_values
            next_token_logits = last_hidden_state[:, -1, :]
            next_token_probs = torch.nn.functional.softmax(next_token_logits)
            topk = torch.topk(next_token_probs, beam, dim=-1)
            probs = topk.values.reshape([batch_size, -1, beam])  # (bs, candidate, beam)
            preds = topk.indices.reshape([batch_size, -1, beam])  # (bs, candidate, beam)

            for j in range(batch_size):
                all_candidates = []
                for n in range(len(sequences[j])):
                    for k in range(beam):
                        seq = sequences[j][n][0]
                        prob = sequences[j][n][1]
                        seq_tmp = seq.copy()
                        seq_tmp.append(preds[j][n][k])
                        candidate = [seq_tmp, prob * probs[j][n][k]]
                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
                sequences[j] = ordered[:beam]

        res = []
        for i in range(batch_size):
            res.append(torch.stack(sequences[i][0][0]))
        res = torch.stack(res)
        return res

    def calculate_loss(self, logit, labels):
        """
        Args:
            preds: torch.FloatTensor, shape=(bs, response_truncate, vocab_size)
            labels: torch.LongTensor, shape=(bs, response_truncate)

        """

        loss = self.loss(logit.reshape(-1, logit.size(-1)), labels.reshape(-1))
        return loss