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

import src.models.extern.deepspeech.model as _model
import src.models.extern.deepspeech.decoder as _decoder

import src.utils.cmd_line as _cmd
import src.utils.utility as _util
import src.data.data_loader as _data_loader


# josephz: Baidu's 'fast' implementation of CTC.
# See https://github.com/baidu-research/warp-ctc
from warpctc_pytorch import CTCLoss

_logger = None

def _getSharedLogger(verbosity=_util.DEFAULT_VERBOSITY):
  global _logger
  if _logger is None:
   _logger = _util.getLogger(os.path.basename(__file__).split('.')[0], verbosity=verbosity)
  return _logger

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def _tensorboard_log(tensorboard_writer, dataset, epoch, loss, wer, cer):
  values = {
    'Avg Train Loss': loss,
    'Avg WER': wer,
    'Avg CER': cer,
  }
  tensorboard_writer.add_scalars(dataset, values, epoch + 1)

def _get_checkpoint_filepath(dataset_dir, num):
  return os.path.join(dataset_dir, "checkpoint_{}.pth".format(num))

def _load_or_create_model(
  epochs,
  dataset,
  continue_from,
  learning_rate,
  rnn_type,
  hidden_size,
  hidden_layers,
  momentum,
  cuda,
  tensorboard_writer
):
  weights_dir = _util.getRelWeightsPath(dataset)
  if continue_from:
    continue_from = _get_checkpoint_filepath(weights_dir, continue_from)
    print('Loading checkpoint model {}'.format(continue_from))

    package = torch.load(continue_from, map_location=lambda storage, loc: storage)
    model = _model.DeepSpeech.load_model_package(package)
    labels = _model.DeepSpeech.get_labels(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    optimizer.load_state_dict(package['optim_dict'])

    # Index start at 0 for training
    start_epoch = int(package.get('epoch', 1)) - 1
    start_iter = package.get('iteration', None)
    if start_iter is None:
      # We saved model after epoch finished, start at the next epoch.
      start_epoch += 1
      start_iter = 0
    else:
      start_iter += 1
    avg_loss = int(package.get('avg_loss', 0))
    loss_results, cer_results, wer_results = package['loss_results'], package['cer_results'], package['wer_results']

    # Previous scores to tensorboard logs
    if tensorboard_writer and package['loss_results'] is not None and start_epoch > 0:
      for i, (loss, wer, cer) in enumerate(zip(package['loss_results'], package['cer_results'], package['wer_results'])):
        _tensorboard_log(tensorboard_writer, dataset, i, loss, wer, cer)
  else:
    avg_loss = start_iter = start_epoch = 0
    loss_results = torch.Tensor(epochs)
    cer_results = torch.Tensor(epochs)
    wer_results = torch.Tensor(epochs)

    with open(os.path.join(weights_dir, 'labels.json')) as label_file:
      labels = str(''.join(json.load(label_file)))

    rnn_type = rnn_type.lower()
    assert rnn_type in _model.supported_rnns, "rnn_type should be either lstm, rnn or gru"

    model = _model.DeepSpeech(
      rnn_hidden_size=hidden_size,
      nb_layers=hidden_layers,
      labels=labels,
      rnn_type=_model.supported_rnns[rnn_type]
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)

  if cuda:
    model.cuda()
  return labels, model, optimizer, (avg_loss, start_iter, start_epoch), (loss_results, cer_results, wer_results)

def _init_averages():
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  return batch_time, data_time, losses

def _get_tensorboard_writer(weights_dir, tensorboard):
  # Initialize weights directory
  os.makedirs(weights_dir, exist_ok=True)

  # josephz: Initialize tensorboard visualization.
  tensorboard_writer = None
  if tensorboard:
    from tensorboardX import SummaryWriter
    tensorboard_writer = SummaryWriter(weights_dir)
  return tensorboard_writer

def _get_datasets(dataset_dir, train_split, labels, batch, num_workers):
  # Load dataset video IDs and shuffle predictably.
  videos = glob.glob(os.path.join(dataset_dir), '*/')
  videos.sort()
  np.random.seed(0)
  np.random.shuffle(videos)

  split_idx = train_split * videos.shape[0]
  train_dataset = _data_loader.FrameCaptionDataset(videos[:split_idx], labels=labels)
  test_dataset = _data_loader.FrameCaptionDataset(videos[split_idx:], labels=labels)

  train_loader = _data.DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=num_workers)
  test_loader = _data.DataLoader(test_dataset, batch_size=batch, num_workers=num_workers)

  return (train_dataset, train_loader), (test_dataset, test_loader)

def train(
  epochs=-1,
  dataset=None,
  batch=-1,
  checkpoint=False,
  train_split=-1.0,
  num_workers=-1,
  hidden_size=-1,
  hidden_layers=-1,
  rnn_type=None,
  cuda=False,
  learning_rate=-1.0,
  momentum=-1.0,
  max_norm=-1,
  annealing=-1.0,
  silent=False,
  tensorboard=False,
  continue_from=-1,
):
  """ Runs the primary training loop.

  :param epochs: Number of epochs to train for.
  :param dataset: Location containing dataset generated by 'generate_dataview'.
  :param batch: Number of sequences that are trained concurrently.
  :param checkpoint: Whether or not to save checpoints for each epoch.
  :param train_split: Fraction of videos which will be in the train set, (1 - train_split) will be validation.
  :param num_workers: Number of workers to use during dataset loading.
  :param hidden_size: Number of hidden units in the RNN.
  :param hidden_layers: Number of hiddel layers in RNN.
  :param rnn_type: Type of RNN cell to use; either rnn, gru, or lstm.
  :param cuda: Use CUDA to train this model.
  :param learning_rate: Initial training learning rate.
  :param momentum: Nesterov SGD momentum.
  :param max_norm: L2 norm cutoff to prevent gradient explosion.
  :param annealing: Annealing applied to learning rate every epoch.
  :param silent: Turn off progress tracking per iteration.
  :param tensorboard: Turn on tensorboard graphing.
  :param continue_from: Checkpoint number to start from.
  """
  weights_dir = _util.getRelWeightsPath(dataset)
  dataset_dir = _util.getRelDatasetsPath(dataset)

  tensorboard_writer = _get_tensorboard_writer(weights_dir, tensorboard)

  labels, model, optimizer, (avg_loss, start_iter, start_epoch), (loss_results, cer_results, wer_results) \
    = _load_or_create_model(epochs, dataset, continue_from, learning_rate, rnn_type, hidden_size, hidden_layers, momentum, cuda, tensorboard_writer)

  (train_dataset, train_loader), (test_dataset, test_loader) = _get_datasets(dataset_dir, train_split, labels, batch, num_workers)

  best_wer = None
  batch_time, data_time, losses = _init_averages()

  print(model)
  print("Number of parameters: %d" % _model.DeepSpeech.get_param_size(model))

  criterion = CTCLoss()
  decoder = _decoder.GreedyDecoder(labels)

  for epoch in range(start_epoch, epochs):
    model.train()
    epoch_start = time.time()

    for i, (data) in enumerate(train_loader, start=start_iter):
      batch_start = time.time()

      inputs, targets, input_percentages, target_sizes = data
      input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

      # Measure elapsed data loading time.
      data_time.update(time.time() - batch_start)

      if cuda:
        inputs = inputs.cuda()

      out, output_sizes = model(inputs, input_sizes)
      out = out.transpose(0, 1)  # TxNxH

      loss = criterion(out, targets, output_sizes, target_sizes)
      # Average loss by minibatch.
      loss /= inputs.size(0)

      loss_value = loss.item()
      if loss_value == np.inf or loss_value == -np.inf:
        print("WARNING: received an inf loss, setting loss value to 0")
        loss_value = 0

      avg_loss += loss_value
      losses.update(loss_value, inputs.size(0))

      # Compute gradient.
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

      # SDG step!
      optimizer.step()

      # Measure elapsed batch time.
      batch_time.update(time.time() - batch_start)
      if not silent:
        print('Epoch[{}][{}{}]'.format(epoch + 1, i + 1, len(train_loader)), end='\t')
        print('Time {.3f} ({.3f})'.format(batch_time.val, batch_time.avg), end='\t')
        print('Data {.3f} ({.3f})'.format(data_time.val, data_time.avg), end='\t')
        print('Loss {.4f} ({.4f})'.format(losses.val, losses.avg))

    avg_loss /= len(train_loader)

    print('Training Summary Epoch: [{}]'.format(epoch + 1), end='\t')
    print('Time taken (s): {.0f}'.format(time.time() - epoch_start))
    print('Time taken (s): {.0f}'.format(time.time() - epoch_start))
    print('Average Loss: {.3f}'.format(avg_loss))

    # Reset start iteration in preparation for next epoch.
    start_iter = 0

    total_cer = total_wer = 0
    model.eval()
    with torch.no_grad():
      for i, (data) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

        # Unflatten targets?
        split_targets = []
        offset = 0
        for size in target_sizes:
          split_targets.append(targets[offset:offset + size])
          offset += size

        if cuda:
          inputs = inputs.cuda()

        out, output_sizes = model(inputs, input_sizes)

        decoded_output, _ = decoder.decode(out.data, output_sizes)
        target_strings = decoder.convert_to_strings(split_targets)

        for x in range(len(target_strings)):
          transcript, reference = decoded_output[x][0], target_strings[x][0]
          total_wer += decoder.wer(transcript, reference) / float(len(reference.split()))
          total_cer += decoder.cer(transcript, reference) / float(len(reference))

      loss_results[epoch] = avg_loss
      wer = wer_results[epoch] = 100 * total_wer / len(test_loader)  # .dataset?
      cer = cer_results[epoch] = 100 * total_cer / len(test_loader)

      print('Validation Summary Epoch: [{}]'.format(epoch + 1), end='\t')
      print('Average WER: {.3f}'.format(wer_results[epoch]), end='\t')
      print('Average CER: {.3f}'.format(cer_results[epoch]), end='\t')

      if tensorboard:
        _tensorboard_log(tensorboard_writer, dataset, epoch + 1, avg_loss, wer, cer)
      if checkpoint:
        weights_path = _get_checkpoint_filepath(dataset_dir, epoch + 1)
        torch.save(
          _model.DeepSpeech.serialize(model, optimizer=optimizer,
            epoch=epoch, loss_results=loss_results, wer_results=wer_results,
            cer_results=cer_results), weights_path)

      # Do annealing.
      optim_state = optimizer.state_dict()
      optim_state['param_groups'][0]['lr'] /= annealing
      optimizer.load_state_dict(optim_state)

      if best_wer is None or best_wer > wer_results[epoch]:
        print('Found better validated model, saving to {}'.format)
        model_path = os.path.join(weights_dir, 'model.pth')
        weights_path = _get_checkpoint_filepath(dataset_dir, epoch + 1)

        if os.path.isfile(weights_path):
          shutil.copyfile(weights_path, model_path)
        else:
          torch.save(
            _model.DeepSpeech.serialize(model, optimizer=optimizer,
              epoch=epoch, loss_results=loss_results, wer_results=wer_results,
              cer_results=cer_results), model_path)
        best_wer = wer
        avg_loss = 0

def main():
  global _logger
  args = _cmd.parseArgsForClassOrScript(train)
  varsArgs = vars(args)
  verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
  _getSharedLogger(verbosity=verbosity).info("Passed arguments: '{}'".format(varsArgs))
  train(**varsArgs)

if __name__ == '__main__':
  main()
