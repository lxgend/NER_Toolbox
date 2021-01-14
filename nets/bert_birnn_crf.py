# coding=utf-8
import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn import LayerNorm
from transformers import BertModel
from torchcrf import CRF
from nets.birnn import BiRNN


class BERT_BiRNN_CRF(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bert_encoder = BertModel.from_pretrained(config.embedding_pretrained)
        self.birnn = BiRNN(config)
        self.crf = CRF(config.num_classes, batch_first=True)

    def forward(self, input_ids, input_mask, label_ids):  # for training
        # mask = xw.data.gt(0).float()

        bert_outputs = self.bert_encoder(input_ids=input_ids, attention_mask=input_mask)
        bert_last_hidden_state = bert_outputs[0]

        lstm_feats = self.birnn(bert_last_hidden_state, input_mask)
        log_likelihood = self.crf(emissions=lstm_feats, tags=label_ids, mask=input_mask)
        # NLL loss
        return (-1) * log_likelihood

    def predict(self, input_ids, input_mask):
        lstm_feats = self.birnn(input_ids, input_mask)
        preds = self.crf.decode(emissions=lstm_feats, mask=input_mask)
        return preds