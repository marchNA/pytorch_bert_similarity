# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.config = config

    def forward(self, x):
        input_ids = x[0].to(self.config.device)
        segment_ids = x[1].to(self.config.device)
        mask_ids = x[2].to(self.config.device)
        _, pooled = self.bert(input_ids=input_ids, attention_mask=mask_ids, token_type_ids=segment_ids)
        out = self.fc(pooled).to(self.config.device)
        return out