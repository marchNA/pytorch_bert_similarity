import time
import torch
import random
import numpy as np
import pandas as pd
from sklearn import metrics
from utils import flat_accuracy, format_time

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import matthews_corrcoef

device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

def train(config, model, train_dataloader, dev_dataloader, test_dataloader):

	# model.cuda()

	optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

	epochs = config.num_epochs

	# Totla number of training steps is number of batches * number of epochs.
	total_steps = len(train_dataloader) * epochs

	# Create the learning rate scheduler.
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps=total_steps)

	# Store the average loss after each epoch so we can plot them.
	loss_values = []

	total_batch = 0 
	dev_best_loss = float('inf')
	last_improve = 0  	# record the num of batch when the dev_dataset' loss reduce
	flag = False  		# record whether the model has no improve for a long time

	for epoch_i in range(0, epochs):
		print("")
		print('=========  Epoch  {:}  /  {:}  ========='.format(epoch_i + 1, epochs))
		print('Training...')

		# Measure how long the traing epoch takes.
		t0 = time.time()

		# Reset the total loss for this epoch.
		total_loss = 0
		
		model.train()

		# For each batch of traing data...
		for step, batch in enumerate(train_dataloader):
			# `batch` contains three pytoch tensor:
			#		[0]: input ids
			#		[1]: segment_ids
			#		[2]: attention masks
			#		[2]: labels
			b_input_ids = batch[0].to(device)
			b_input_segment = batch[1].to(device)
			b_input_mask = batch[2].to(device)
			b_labels = batch[3].to(device)

			model.zero_grad()
	
			outputs = model(b_input_ids, token_type_ids=b_input_segment, attention_mask=b_input_mask, labels=b_labels)

			# loss and logits (logits contain the probabilties of two classes)
			loss, logit = outputs[:2]

			# Process update every 100 batches.
			if step % 100 == 0 and not step == 0:
				true = b_labels.data.cpu()
				predic = torch.max(logit.data, 1)[1].cpu()		# get the max one
				train_acc = metrics.accuracy_score(true, predic)
				dev_acc, dev_loss = evaluate(config, model, dev_dataloader)

				# Calculate elapsed time in minutes.
				elapsed = format_time(time.time() - t0)

				if dev_loss < dev_best_loss:
					torch.save(model.state_dict(), config.save_path)
					improve = '*'
					last_improve = total_batch
				else:
					improve = ''

				# Report progress.
				msg = 'Batch {0:>5,} of {1:>5,}, Train Loss: {2:>5.2},  Train Acc: {3:>6.2},  Val Loss: {4:6.2}, Val Acc: {5:>6.2},  Elapsed: {6}  {7}.'
				print(msg.format(step, len(train_dataloader), loss.item(), train_acc,  dev_loss, dev_acc, elapsed, improve))
			
			total_batch += 1

			# Accumulate the training loss over all of the batches so that we can
			# calculate the average loss at the end. `loss` is a Tensor containg a
			# single value: the `.item()` function just returns the Python value
			# from the tensor.
			total_loss += loss.item()

			# Perform a backward pass to calculate the gradients.
			loss.backward()

			# Clip the norm of the gradients to 1.0.
			# This is to help prevent the "exploding gradients" problem.
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

			optimizer.step()

			# Update the learning rate.
			scheduler.step()

		# Calculate the average loss over the training data.
		avg_train_loss = total_loss / len(train_dataloader)

		# Store the loss value for plotting the learning curve.
		loss_values.append(avg_train_loss)

		print("")
		print("Average training loss:	{0:.2f}".format(avg_train_loss))
		print("Time usage of A epoch: {:}".format(format_time(time.time() - t0)))


	print("")
	print("Training complete!")
	print("Total time usage: {:}".format(format_time(time.time() - t0)))
	test(config, model, test_dataloader)

def evaluate(config, model, dataloader, test=False):
	"""
	Evaluate the model in dev dataset
	params:
	:model 			bert model
	:dataloader		Dataloader

	return:
	: average accuracy
	: average loss
	"""

	# Tracking variables
	eval_loss = 0, 
	predict_all = np.array([], dtype=int)
	labels_all = np.array([], dtype=int)

	# Evaluate data for one epoch
	for batch in dataloader:

		# Add batch to GPU
		batch = tuple(t.to(device) for t in batch)

		# Unpack the inputs from our dataloaders
		b_input_ids, b_input_segments, b_input_mask, b_labels = batch

		# Telling the model not to compute or store gradients, saving memory and
		# spedding up validation
		with torch.no_grad():
			outputs = model(b_input_ids, token_type_ids=b_input_segments, attention_mask=b_input_mask, labels=b_labels)

		# Get the "logits" output by the model. The "logits" are the output
		# values prior to applying an activation function like the softmax.
		loss, logits = outputs[:2]

		# Move logits and labels to CPU
		predicts = logits.detach().cpu().numpy()
		labels = b_labels.to('cpu').numpy()

		predicts, labels = flat_accuracy(predicts, labels)


		labels_all = np.append(labels_all, labels)
		predict_all = np.append(predict_all, predicts)

		# # Calculate the accuracy for this batch of test sentences.
		# tmp_eval_accuracy = flat_accuracy(predic, labels_ids)

		# # Accumulate the total accuracy.
		# predict_all = np.append(predict_all, tmp_eval_accuracy)

		eval_loss += loss.cpu().numpy()
	acc = metrics.accuracy_score(labels_all, predict_all)
	if test:
		report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
		confusion = metrics.confusion_matrix(labels_all, predict_all)
		return acc, eval_loss[0] / len(dataloader), report, confusion
	return acc, eval_loss[0] / len(dataloader)


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