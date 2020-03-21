import os
import torch
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def get_examples(data_dir, filename):
    file_path = os.path.join(data_dir, filename)
    df = pd.read_csv(file_path, encoding='utf-8')
    datas = []
    for index, data in enumerate(df.values):
        text_a = str(data[0]) # 转换编码格式
        text_b = str(data[1])
        label = str(data[2])
        datas.append([[text_a, text_b], label])
    return datas

def get_sent(data_path):
	train_sent = get_examples(data_path, 'train.csv')
	dev_sent = get_examples(data_path, 'dev.csv')
	test_sent = get_examples(data_path, 'test.csv')
	return train_sent, dev_sent, test_sent

def get_demo_sent(data_path):
	train_sent = get_examples(data_path, 'train_demo.csv')
	dev_sent = get_examples(data_path, 'dev_demo.csv')
	test_sent = get_examples(data_path, 'test_demo.csv')
	return train_sent, dev_sent, test_sent

def get_encode(tokenizer, text_a, text_b):
	print("Getting sentences pair encode.")
	encoded_pair = tokenizer.encode(text_a, text_b, add_special_tokens=True)
	return encoded_pair

def get_input_ids(tokenizer, data):
	input_ids = []
	print("")
	print("Getting input_ids")
	try:
		with tqdm(data) as t:
			for sent in t:
				encoded_sent = tokenizer.encode(
					sent[0][0],
					sent[0][1],
					add_special_tokens = True,
				)
				input_ids.append(encoded_sent)
	except KeyboardInterrupt:
		t.close()
		raise
	t.close
	return input_ids

def get_mask(input_ids):
	# Create attention masks
	attention_masks = []
	print("")
	print("Getting attention masks")
	# For each sentence...
	try:
		with tqdm(input_ids) as t:
			for sent in t:
				
				# Create the attention mask.
				#   - If a token ID is 0, then it's padding, set the mask to 0.
				#   - If a token ID is > 0, then it's a real token, set the mask to 1.
				att_mask = [int(token_id > 0) for token_id in sent]
				
				# Store the attention mask for this sentence.
				attention_masks.append(att_mask)
	except KeyboardInterrupt:
		t.close()
		raise
	t.close		
	return torch.tensor(attention_masks).long()

def get_segment_ids(input_ids):
	segment_ids = []
	print("")
	print("Getting segment ids")
	try:
		with tqdm(input_ids) as t:
			for input_id in t:
				SEP_flag = input_id.index(102)
				segment_id = []
				for index, seg in enumerate(input_id):
					if index < SEP_flag:
						segment_id.append(0)
					else:
						segment_id.append(1)
				segment_ids.append(segment_id)
	except KeyboardInterrupt:
		t.close()
		raise
	t.close
	return segment_ids

def get_label(data):
	return torch.tensor([int(sent[1]) for sent in data]).long()

def padding(config, input_ids):
	return torch.tensor(pad_sequences(input_ids, maxlen=config.max_len, dtype="long", 
	                         value=0, truncating="post", padding="post")).long()

def get_input(config, input_sent):
	input_ids = get_input_ids(config.tokenizer, input_sent)
	segment_ids = get_segment_ids(input_ids)

	input_ids = padding(config, input_ids)
	segment_ids = padding(config, segment_ids)

	input_masks = get_mask(input_ids)
	label_ids = get_label(input_sent)
	return input_ids, segment_ids, input_masks, label_ids

def get_data_loader(config, data_sent):
    input_ids, segment_ids, mask_ids, lable_ids = get_input(config, data_sent)

    data = TensorDataset(input_ids, segment_ids, mask_ids, lable_ids)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=config.batch_size)  
    return dataloader

# Funcion to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
	pred_flat = np.argmax(preds, axis=1).flatten()
	labels_flat = labels.flatten()
	return pred_flat, labels_flat

def format_time(elapsed):
	'''
	Takes a time in seconds and returns a string hh:mm:ss
	'''
	# Round to the nearest second.
	elapsed_rounded = int(round(elapsed))

	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds=elapsed_rounded))


# memory footprint support libraries/code
# !ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
# !pip install gputil
# !pip install psutil
# !pip install humanize

def printm(config):
	if config.device == 'cuda':
		import psutil
		import humanize
		import os
		import GPUtil as GPU

		GPUs = GPU.getGPUs()
		# XXX: only one GPU on Colab and isn’t guaranteed
		gpu = GPUs[0]

		process = psutil.Process(os.getpid())
		print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available), " |     Proc size: " + humanize.naturalsize(process.memory_info().rss))
		print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total     {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
		return gpu.memoryUtil
	else:
		return 1