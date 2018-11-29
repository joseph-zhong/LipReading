#!/usr/bin/env python3

import os
import glob
import json
import torch
import torch.utils.data as _data

import src.utils.utility as _util
import src.data.data_loader as _data_loader
import src.scripts.train_better_model as _train

import src.models.lipreader.better_model as _better_model

# Init Data.
batch=4
num_workers=1
dataset="StephenColbert/nano1"
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
dataset = _data_loader.FrameCaptionSentenceDataset(vid_id_dirs, labels, batch_size=batch)
data_loader = _data.DataLoader(dataset, batch_size=batch, num_workers=num_workers,
  collate_fn=_data_loader._collate_sentences_fn)

# Init Models.
frame_dim = 68 * 3
hidden_size = 700
char_dim = 300
encoder = _better_model.VideoEncoder(frame_dim, hidden_size,
  rnn_type='LSTM', num_layers=1, bidirectional=True,
  rnn_dropout=0)
decoding_step = _better_model.CharDecodingStep(encoder, char_dim=char_dim, output_size=len(dataset.char2idx),
  char_padding_idx=_data_loader._markers2Id[_data_loader.PAD],
  rnn_dropout=0)

# Train.
learning_rate = 3e-4
_train.train(encoder, decoding_step, data_loader,
  opt=torch.optim.Adam(list(encoder.parameters()) + list(decoding_step.parameters()), lr=learning_rate),
  device=torch.device('cpu'), # device=torch.device('cuda'),
  char2idx=dataset.char2idx,
  teacher_forcing_ratio=1,
  grad_norm=None)