import os
import torch
from transformers import BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, data_path, bert_path):
        self.model_name = 'bert'
        self.save_path = data_path + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.class_list = ['0', '1']

        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 4                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.max_len = 64                                               # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-4                                       # 学习率
        self.bert_path = bert_path
        print('Loading BERT tokenizer...')	
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
