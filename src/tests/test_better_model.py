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
dataset="StephenColbert/118"
sentence_dataset=False
threshold=0.8
weights_dir = _util.getRelWeightsPath(dataset)
dataset_dir = _util.getRelDatasetsPath(dataset)
raw_dir = _util.getRelRawPath(dataset)
vid_id_dirs = glob.glob(os.path.join(dataset_dir, '*/'))
vid_id_dirs.sort()

## Init dataset and dataset loader.
train_dataset = _data_loader.FrameCaptionDataset(dataset, 'train', vid_id_dirs, labels='labels.json', sentence_dataset=sentence_dataset)

train_data_loader = _data.DataLoader(train_dataset, batch_size=batch, num_workers=num_workers,
  collate_fn=_data_loader._collate_fn)
print("train_dataset len:", len(train_dataset))

batch=4
num_workers=1
test_dataset="StephenColbert/nano"
weights_dir = _util.getRelWeightsPath(test_dataset)
dataset_dir = _util.getRelDatasetsPath(test_dataset)
raw_dir = _util.getRelRawPath(test_dataset)
vid_id_dirs = glob.glob(os.path.join(dataset_dir, '*/'))
vid_id_dirs.sort()

## Init dataset and dataset loader.
test_dataset = _data_loader.FrameCaptionDataset(test_dataset, 'test', vid_id_dirs, labels='labels.json', sentence_dataset=sentence_dataset)

test_data_loader = _data.DataLoader(test_dataset, batch_size=batch, num_workers=num_workers,
  collate_fn=_data_loader._collate_fn)
print("test_dataset len:", len(test_dataset))
assert train_dataset.char2idx == test_dataset.char2idx

# Init Models.
cuda = True
device = torch.device('cuda') if cuda else torch.device('cpu')
frame_dim = 68 * 3
hidden_size = 700
char_dim = 300
encoder = _better_model.VideoEncoder(frame_dim, hidden_size,
  rnn_type='LSTM', num_layers=1, bidirectional=True, rnn_dropout=0,
  enable_ctc=True, vocab_size=len(train_dataset.char2idx), char2idx=train_dataset.char2idx, device=device).to(device)
decoding_step = _better_model.CharDecodingStep(encoder,
  char_dim=char_dim, vocab_size=len(train_dataset.char2idx),
  char2idx=train_dataset.char2idx,
  rnn_dropout=0, attention_type='1_layer_nn', attn_hidden_size=-1, device=device).to(device)

# Train.
learning_rate = 3e-4
# opt = torch.optim.Adam(list(encoder.parameters()) + list(decoding_step.parameters()), lr=learning_rate)

_train.eval(encoder, decoding_step, test_data_loader, device, train_dataset.char2idx)

for i in range(50):
  print(f'epoch {i}')
  _train.train(encoder, decoding_step, train_data_loader,
    # opt=opt,
    opt=torch.optim.Adam(list(encoder.parameters()) + list(decoding_step.parameters()), lr=learning_rate),
    device=device,
    char2idx=train_dataset.char2idx,
    teacher_forcing_ratio=1,
    grad_norm=50)
  _train.eval(encoder, decoding_step, test_data_loader, device, train_dataset.char2idx)
