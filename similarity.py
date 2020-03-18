import os
import random
from args import Config

from train_eval import train

from utils import get_sent, get_input, get_data_loader, printm

data_path = './data'
# data_path = './data'

bert_path = './bert'
# bert_path = '/content/bert-utils/bert/multilingual_L-12_H-768_A-12'

if __name__ =='__main__':
	config = Config(data_path, bert_path)
	if config.device =='cuda':
		memoryUtil = printm(config)
		if memoryUtil > 0.2:
			try:
				sys.exit()
			except:
				print('MemoryUtil FULL')

	train_sent, dev_sent, test_sent = get_sent(data_path)

	train_dataloader = get_data_loader(config, train_sent)
	dev_dataloader = get_data_loader(config, dev_sent)
	test_dataloader = get_data_loader(config, test_sent)

	model = config.model

	train(config, model, train_dataloader, dev_dataloader, test_dataloader)

