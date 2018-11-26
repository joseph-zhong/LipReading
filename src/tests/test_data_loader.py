#!/usr/bin/env python3
"""

"""

import os
import json
import glob

import torch
import torch.utils.data as _data

import src.utils.utility as _util
import src.data.data_loader as _data_loader

import src.scripts.train_model as _train

dataset="StephenColbert/micro"
batch=4
num_workers=1
hidden_size=800
hidden_layers=5
train_split=0.5
rnn_type="gru"
epochs=70
cuda=False
learning_rate=3e-4
momentum=0.9
max_norm=400
anneal=1.1
silent=True
checkpoint=True
tensorboard=True
continue_from=0

weights_dir = _util.getRelWeightsPath(dataset)
dataset_dir = _util.getRelDatasetsPath(dataset)
raw_dir = _util.getRelRawPath(dataset)

# tensorboard_writer = _train._get_tensorboard_writer(weights_dir, tensorboard)
#
# labels, model, optimizer, \
#   avg_tr_loss, avg_val_loss, start_iter, start_epoch, \
#   loss_results, val_loss_results, cer_results, wer_results = _train._load_or_create_model(
#     epochs, dataset, continue_from, learning_rate, rnn_type, hidden_size, hidden_layers, momentum, cuda, tensorboard_writer)
#
# (train_dataset, train_loader), (test_dataset, test_loader) = _train._get_datasets(
#   dataset_dir, train_split, labels, batch, num_workers)
#
# for i, (data) in enumerate(train_loader, start=start_iter):
#   assert len(data) == 2
#   inp, label = data
#   print("[i='{}'] [inp.shape='{}'] [type(label)='{}']".format(i, inp.shape, type(label)))

vid_id_dirs = glob.glob(os.path.join(dataset_dir, '*'))
vid_id_dirs.sort()

labels_path = os.path.join(raw_dir, 'labels.json')
print("Loading labels_path", labels_path)
with open(labels_path, 'r') as fin:
  labels = json.load(fin)
print("init:")
dataset = _data_loader.FrameCaptionSentenceDataset(vid_id_dirs, labels, batch_size=batch)
train_loader = _data.DataLoader(dataset, batch_size=batch, num_workers=num_workers,
  collate_fn=_data_loader._collate_sentences_fn)

for data in train_loader:
  frames, caption, inp, label, inp_len, label_len = data
  print('frames')




