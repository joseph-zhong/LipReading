#!/usr/bin/env python3

import os
import glob
import json
import torch
import torch.utils.data as _data

import src.utils.utility as _util
import src.data.data_loader as _data_loader
import src.train.train_better_model as _train

import src.models.lipreader.better_model as _better_model

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

# Init Data.
batch=4
num_workers=1
dataset="StephenColbert/micro_no_vtx"
sentence_dataset=False
threshold=0.8
weights_dir = _util.getRelWeightsPath(dataset)
dataset_dir = _util.getRelDatasetsPath(dataset)
raw_dir = _util.getRelRawPath(dataset)
vid_id_dirs = glob.glob(os.path.join(dataset_dir, '*'))
vid_id_dirs.sort()

## Load alphabet.
labels_path = os.path.join(raw_dir, 'labels.json')
print("Loading labels_path", labels_path)
with open(labels_path, 'r') as fin:
  labels = json.load(fin)

## Init dataset and dataset loader.
if sentence_dataset:
  train_dataset = _data_loader.FrameCaptionSentenceDataset(vid_id_dirs, labels)
else:
  train_dataset = _data_loader.FrameCaptionDataset(vid_id_dirs, labels)

train_data_loader = _data.DataLoader(train_dataset, batch_size=batch, num_workers=num_workers,
  collate_fn=_data_loader._collate_fn)
print("train_dataset len:", len(train_dataset))

batch=4
num_workers=1
test_dataset="StephenColbert/nano"
weights_dir = _util.getRelWeightsPath(test_dataset)
dataset_dir = _util.getRelDatasetsPath(test_dataset)
raw_dir = _util.getRelRawPath(test_dataset)
vid_id_dirs = glob.glob(os.path.join(dataset_dir, '*'))
vid_id_dirs.sort()

## Load alphabet.
labels_path = os.path.join(raw_dir, 'labels.json')
print("Loading labels_path", labels_path)
with open(labels_path, 'r') as fin:
  labels = json.load(fin)

## Init dataset and dataset loader.
if sentence_dataset:
  test_dataset = _data_loader.FrameCaptionSentenceDataset(vid_id_dirs, labels)
else:
  test_dataset = _data_loader.FrameCaptionDataset(vid_id_dirs, labels)

test_data_loader = _data.DataLoader(test_dataset, batch_size=batch, num_workers=num_workers,
  collate_fn=_data_loader._collate_fn)
print("test_dataset len:", len(test_dataset))
assert train_dataset.char2idx == test_dataset.char2idx

# Init Models.
frame_dim = 68 * 3
hidden_size = 700
char_dim = 300
encoder = _better_model.VideoEncoder(frame_dim, hidden_size,
  rnn_type='LSTM', num_layers=1, bidirectional=True, rnn_dropout=0,
  enable_ctc=True, vocab_size=len(train_dataset.char2idx), char2idx=train_dataset.char2idx)
decoding_step = _better_model.CharDecodingStep(encoder,
  char_dim=char_dim, vocab_size=len(train_dataset.char2idx),
  char2idx=train_dataset.char2idx,
  rnn_dropout=0, attention_type='1_layer_nn')

# Train.
learning_rate = 3e-4
# opt = torch.optim.Adam(list(encoder.parameters()) + list(decoding_step.parameters()), lr=learning_rate)

_train.eval(encoder, decoding_step, test_data_loader, torch.device('cpu'), train_dataset.char2idx)

for i in range(50):
  print(f'epoch {i}')
  _train.train(encoder, decoding_step, train_data_loader,
    # opt=opt,
    opt=torch.optim.Adam(list(encoder.parameters()) + list(decoding_step.parameters()), lr=learning_rate),
    device=torch.device('cpu'), # device=torch.device('cuda'),
    char2idx=train_dataset.char2idx,
    teacher_forcing_ratio=1,
    grad_norm=50)
  _train.eval(encoder, decoding_step, test_data_loader, torch.device('cpu'), train_dataset.char2idx)
