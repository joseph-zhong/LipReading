#!/usr/bin/env python3
"""
train.py
---

This trains a specified model.

"""
import os
import time
import json
import torch
import torch.utils.data as _data

import numpy as np
import glob
import shutil
import tqdm

import src.utils.cmd_line as _cmd
import src.utils.utility as _util

import src.data.data_loader as _data_loader
import src.train.train_better_model as _train
import src.models.lipreader.better_model as _better_model

_logger = None

def _getSharedLogger(verbosity=_util.DEFAULT_VERBOSITY):
  global _logger
  if _logger is None:
   _logger = _util.getLogger(os.path.basename(__file__).split('.')[0], verbosity=verbosity)
  return _logger

def _get_datasets(dataset_name, train_split, sentence_dataset,
    threshold=0.8,
    labels='labels.json', rand=None, refresh=False):

  # REVIEW josephz: If we can load from pickles, we should not even do this split. We could have a helper factory thingy?
  # Load dataset video IDs and shuffle predictably.
  train_ids, val_ids, test_ids = _data_loader.split_dataset(dataset_name, train_split=train_split, rand=rand)
  train_dataset = _data_loader.FrameCaptionSentenceDataset(os.path.join(dataset_name, 'train'), train_ids, labels,
    threshold=threshold, sentence_dataset=sentence_dataset, refresh=refresh)
  val_dataset = _data_loader.FrameCaptionSentenceDataset(os.path.join(dataset_name, 'val'), val_ids, labels,
    threshold=threshold, sentence_dataset=sentence_dataset, refresh=refresh)
  test_dataset = _data_loader.FrameCaptionSentenceDataset(os.path.join(dataset_name, 'test'), test_ids, labels,
    threshold=threshold, sentence_dataset=sentence_dataset, refresh=refresh)

  print()
  print("Dataset Information:")
  print("\tTrain Dataset Size:", len(train_dataset))
  print("\tVal Dataset Size:", len(val_dataset))
  print("\tTest Dataset Size:", len(test_dataset))
  print()
  return train_dataset, val_dataset, test_dataset

def _init_models(
    char2idx,
    num_layers,
    frame_dim,
    hidden_size,
    char_dim,
    enable_ctc,

    rnn_type,
    attention_type,
    bidirectional,
    rnn_dropout,
):
  encoder = _better_model.VideoEncoder(frame_dim, hidden_size,
    rnn_type=rnn_type, num_layers=num_layers, bidirectional=bidirectional, rnn_dropout=rnn_dropout,
    enable_ctc=enable_ctc, vocab_size=len(char2idx), char2idx=char2idx)
  decoding_step = _better_model.CharDecodingStep(encoder,
    char_dim=char_dim, vocab_size=len(char2idx), char2idx=char2idx, rnn_dropout=rnn_dropout, attention_type=attention_type)

  return encoder, decoding_step

def train(
    data="StephenColbert/medium_no_vtx1",
    labels="labels.json",
    sentence_dataset=True,
    occlussion_threshold=0.8,
    train_split=0.8,
    num_workers=1,
    refresh=False,

    num_epochs=50,
    batch_size=4,
    learning_rate=1e-2,
    enable_ctc=False,
    teacher_forcing_ratio=1.0,
    grad_norm=50,

    num_layers=1,
    frame_dim=68*3,
    hidden_size=700,
    char_dim=300,

    rnn_type='LSTM',
    attention_type='1_layer_nn',
    bidirectional=False,
    rnn_dropout=0.0,

    seed=123456,
    cuda=False,
):
  """ Runs the primary training loop.

  :param data:
  :param labels:
  :param sentence_dataset:
  :param occlussion_threshold:
  :param train_split:
  :param num_workers:
  :param num_epochs:
  :param batch_size:
  :param learning_rate:
  :param enable_ctc:
  :param teacher_forcing_ratio:
  :param grad_norm:
  :param num_layers:
  :param frame_dim:
  :param hidden_size:
  :param char_dim:
  :param rnn_type:
  :param attention_type:
  :param bidirectional:
  :param rnn_dropout:
  :param seed:
  :param cuda:
  """
  # Setup seed.
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  rand = np.random.RandomState(seed=seed)

  # Setup device.
  # REVIEW josephz: Is there a clean way to use multiple or different GPUs?
  device = torch.device('cuda') if cuda else torch.device('cpu')

  # Init Data.
  train_dataset, val_dataset, test_dataset = _get_datasets(data, train_split, sentence_dataset,
    threshold=occlussion_threshold, labels=labels, rand=rand, refresh=refresh)
  train_loader = _data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_data_loader._collate_fn)
  val_loader = _data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_data_loader._collate_fn)
  test_loader = _data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_data_loader._collate_fn)

  # Init Models.
  encoder, decoding_step = _init_models(train_dataset.char2idx, num_layers, frame_dim, hidden_size, char_dim,
    enable_ctc, rnn_type, attention_type, bidirectional, rnn_dropout)

  # Train.
  val_cers = []
  test_cers = []
  train_decoder_losses = []
  train_ctc_losses = []

  ts = time.time()
  for i in range(num_epochs):
    decoder_loss, correct, count = _train.eval(encoder, decoding_step, val_loader, device, train_dataset.char2idx)
    val_cer = (count - correct).float() / count

    avg_decoder_loss, avg_ctc_loss = _train.train(encoder, decoding_step, train_loader,
      opt=torch.optim.Adam(list(encoder.parameters()) + list(decoding_step.parameters()), lr=learning_rate),
      device=device,
      char2idx=train_dataset.char2idx,
      teacher_forcing_ratio=teacher_forcing_ratio,
      grad_norm=grad_norm)

    decoder_loss, correct, count = _train.eval(encoder, decoding_step, test_loader, device, train_dataset.char2idx)
    test_cer = (count - correct).float() / count

    val_cers.append(val_cer)
    test_cers.append(test_cer)
    train_decoder_losses.append(avg_decoder_loss)
    avg_ctc_loss.append(avg_ctc_loss)
  te = time.time()
  total_time = te - ts
  print()
  print("Training complete: Took '{}' seconds, or '{}' per epoch".format(total_time, total_time / num_epochs))
  print("Training Statistics")
  print("\tBest Val CER: '{}'".format(np.min(val_cers)))
  print("\tBest Test CER: '{}'".format(np.min(test_cers)))
  print("\tBest Decoder Loss: '{}'".format(np.min(train_decoder_losses)))
  print("\tBest CTC Loss: '{}'".format(np.min(train_ctc_losses)))
  print()

def main():
  global _logger
  args = _cmd.parseArgsForClassOrScript(train)
  varsArgs = vars(args)
  verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
  _getSharedLogger(verbosity=verbosity).info("Passed arguments: '{}'".format(varsArgs))
  train(**varsArgs)

if __name__ == '__main__':
  main()
