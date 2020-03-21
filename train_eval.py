#!/usr/bin/python
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from sklearn import metrics
from utils import flat_accuracy, format_time

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import matthews_corrcoef

def train(config, model, train_dataloader, dev_dataloader, test_dataloader):

	start_time = time.time()
	
	# model.cuda()
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
	
	optimizer = AdamW(optimizer_grouped_parameters, lr = 2e-5, eps = 1e-8)

	# Create the learning rate scheduler.
	total_steps = len(train_dataloader) * config.num_epochs
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps=total_steps)


	total_batch = 0  # 记录进行到多少batch
	dev_best_loss = float('inf')
	last_improve = 0  # 记录上次验证集loss下降的batch数
	flag = False  # 记录是否很久没有效果提升
	model.train()

	for epoch in range(0, config.num_epochs):
		# For each batch of traing data...
		print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
		for i, trains in enumerate(train_dataloader):
			# `batch` contains three pytoch tensor:
			#		[0]: input ids
			#		[1]: segment_ids
			#		[2]: attention masks
			#		[3]: labels
			input_ids = trains[0].to(config.device)
			segment_ids = trains[1].to(config.device)
			mask_ids = trains[2].to(config.device)
			labels = trains[3].to(config.device)

			outputs = model(input_ids=input_ids, attention_mask=mask_ids, token_type_ids=segment_ids, labels=labels)
			model.zero_grad()
			loss, logit = outputs	# outputs[0]: loss, outputs[1]: predict  (batch_size, num_classes)

			loss.backward()

			optimizer.step()
			scheduler.step()
			if total_batch % 100 == 0:
				# 每多少轮输出在训练集和验证集上的效果
				true = labels.data.cpu()
				predic = torch.max(logit.data, 1)[1].cpu()
				train_acc = metrics.accuracy_score(true, predic)
				dev_acc, dev_loss = evaluate(config, model, dev_dataloader)
				if dev_loss < dev_best_loss:
					dev_best_loss = dev_loss
					torch.save(model.state_dict(), config.save_path)
					improve = '*'
					last_improve = total_batch
				else:
					improve = ''
				time_dif = format_time(time.time() - start_time)
				msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
				print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
				model.train()
			total_batch += 1
			if total_batch - last_improve > config.require_improvement:
				# 验证集loss超过1000batch没下降，结束训练
				print("No optimization for a long time, auto-stopping...")
				flag = True
				break
		if flag:
			break

	print("")
	print("Training complete!")
	print("Total time usage: {:}".format(format_time(time.time() - start_time)))
	test(config, model, test_dataloader)

def test(config, model, dataloader):
	# test
	model.load_state_dict(torch.load(config.save_path))
	model.eval()
	print('Predicting labels for {:,} test sentences...'.format(len(dataloader)))
	test_acc, test_loss, test_report, test_confusion = evaluate(config, model, dataloader, test=True)
	t0 = time.time()
	msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
	print(msg.format(test_loss, test_acc))
	print("Precision, Recall and F1-Score...")
	print(test_report)
	print("Confusion Matrix...")
	print(test_confusion)
	elapsed = format_time(time.time() - t0)
	print("Time usage:", elapsed)

def evaluate(config, model, dataloader, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
		for i, dev in enumerate(dataloader):
			# `batch` contains three pytoch tensor:
			#		[0]: input ids
			#		[1]: segment_ids
			#		[2]: attention masks
			#		[3]: labels
			input_ids = dev[0].to(config.device)
			segment_ids = dev[1].to(config.device)
			mask_ids = dev[2].to(config.device)
			labels = dev[3].to(config.device)

			outputs = model(input_ids=input_ids, attention_mask=mask_ids, token_type_ids=segment_ids, labels=labels)
			loss, logit = outputs[:2]

			labels = labels.data.cpu().numpy()
			predic = torch.max(logit.data, 1)[1].cpu().numpy()
			labels_all = np.append(labels_all, labels)
			predict_all = np.append(predict_all, predic)
			loss_total += loss.item()

    acc = metrics.accuracy_score(labels_all, predict_all)

	if test:
		report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
		confusion = metrics.confusion_matrix(labels_all, predict_all)
		return acc, loss_total / len(dataloader), report, confusion
	return acc, loss_total / len(dataloader)