import os
import random
from args import Config
import time
from train_eval import train
from model.bert import Model
from utils import get_sent, get_input, get_data_loader, printm, format_time


data_path = './data'

bert_path = './bert'

if __name__ =='__main__':
	config = Config(data_path, bert_path)
	memoryUtil = printm(config)
  	print(memoryUtil)	
	if memoryUtil > 0.2:
		try:
			os._exit(0)
		except:
			print('MemoryUtil FULL')

	start_time = time.time()
	print("Loading data...")
	train_sent, dev_sent, test_sent = get_sent(data_path)
	train_dataloader = get_data_loader(config, train_sent)
	dev_dataloader = get_data_loader(config, dev_sent)
	test_dataloader = get_data_loader(config, test_sent)
	print("Time usage:", format_time(time.time() - start_time))

	# train
	model = Model(config)
	train(config, model, train_dataloader, dev_dataloader, test_dataloader)

