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

import src.models.lipreader.model as _model
import src.models.lipreader.decoder as _decoder

import src.utils.cmd_line as _cmd
import src.utils.utility as _util
import src.data.data_loader as _data_loader

_logger = None
_labels = [" ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", ">", "?", "@", "[", "]", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

def _getSharedLogger(verbosity=_util.DEFAULT_VERBOSITY):
  global _logger
  if _logger is None:
   _logger = _util.getLogger(os.path.basename(__file__).split('.')[0], verbosity=verbosity)
  return _logger

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

def _tensorboard_log(tensorboard_writer, dataset, epoch, loss, wer, cer, mode="Train"):
  values = {
    'Avg {} Loss'.format(mode): loss,
    'Avg {} WER'.format(mode): wer,
    'Avg {} CER'.format(mode): cer,
  }
  _getSharedLogger().debug("Writing Tensorboard log: '%s' for epoch: '%d'", values, epoch + 1)
  tensorboard_writer.add_scalars(dataset, values, epoch + 1)

def _get_checkpoint_filepath(dataset_dir, num):
  res = os.path.join(dataset_dir, "ckpts", "checkpoint_{}.pth".format(num))
  _util.mkdirP(os.path.dirname(res))
  return res

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
    model = _model.LipReader.load_model_package(package)
    labels = _model.LipReader.get_labels(model)

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
    avg_tr_loss = int(package.get('avg_tr_loss', 0))
    avg_val_loss = int(package.get('avg_val_loss', 0))
    tr_loss_results = package['tr_loss_results']
    val_loss_results = package['val_loss_results']
    cer_results = package['cer_results']
    wer_results = package['wer_results']

    # Previous scores to tensorboard logs
    if tensorboard_writer and package['tr_loss_results'] is not None and start_epoch > 0:
      # REVIEW josephz: Also include train?
      # package['tr_loss_results']
      for i, (val_loss, wer, cer) in enumerate(zip(package['val_loss_results'],
          package['val_cer_results'],
          package['val_wer_results'])):
        _tensorboard_log(tensorboard_writer, dataset, i, val_loss, wer, cer, mode="Validation")
  else:
    avg_tr_loss = avg_val_loss = start_iter = start_epoch = 0
    tr_loss_results = torch.Tensor(epochs)
    val_loss_results = torch.Tensor(epochs)
    cer_results = torch.Tensor(epochs)
    wer_results = torch.Tensor(epochs)

    labels_path = os.path.join(weights_dir, 'labels.json')
    try:
      with open(labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))
    except:
      labels = _labels
      _getSharedLogger().warning("Could not open '{}'... using hardcoded labels: '{}'".format(labels_path, labels))

    rnn_type = rnn_type.lower()
    assert rnn_type in _model.supported_rnns, "rnn_type should be either lstm, rnn or gru"

    model = _model.LipReader(
      rnn_hidden_size=hidden_size,
      nb_layers=hidden_layers,
      labels=labels,
      rnn_type=_model.supported_rnns[rnn_type]
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)

  if cuda:
    model.cuda()
  return labels, model, optimizer, \
    (avg_tr_loss, avg_val_loss, start_iter, start_epoch), \
    (tr_loss_results, val_loss_results, cer_results, wer_results)

def _init_averages():
  batch_time = AverageMeter()
  data_time = AverageMeter()
  tr_losses = AverageMeter()
  val_losses = AverageMeter()
  return batch_time, data_time, tr_losses, val_losses

def _get_tensorboard_writer(weights_dir, tensorboard):
  # Initialize weights directory
  os.makedirs(weights_dir, exist_ok=True)

  # josephz: Initialize tensorboard visualization.
  tensorboard_writer = None
  if tensorboard:
    from tensorboardX import SummaryWriter
    tensorboard_writer = SummaryWriter(weights_dir)
    _getSharedLogger().info("Writing Tensorboard logs to '%s'", weights_dir)
  else:
    _getSharedLogger().warning("Tensorboard disabled...")
  return tensorboard_writer

def _get_datasets(dataset_dir, train_split, labels, batch, num_workers):
  # Load dataset video IDs and shuffle predictably.
  videos = glob.glob(os.path.join(dataset_dir, '*'))
  videos.sort()
  np.random.seed(0)
  np.random.shuffle(videos)

  split_idx = int(train_split * len(videos))
  train_dataset = _data_loader.FrameCaptionDataset(videos[:split_idx], labels=labels)
  test_dataset = _data_loader.FrameCaptionDataset(videos[split_idx:], labels=labels)

  train_sampler = _data_loader.BucketingSampler(train_dataset, batch_size=batch)
  train_loader = _data.DataLoader(train_dataset, batch_size=batch, num_workers=num_workers, collate_fn=_data_loader._collate_fn)
  test_loader = _data.DataLoader(test_dataset, batch_size=batch, num_workers=num_workers, collate_fn=_data_loader._collate_fn)

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
  seed=123456,
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
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  weights_dir = _util.getRelWeightsPath(dataset)
  dataset_dir = _util.getRelDatasetsPath(dataset)

  tensorboard_writer = _get_tensorboard_writer(weights_dir, tensorboard)

  # REVIEW josephz: Can this be further broken down?
  labels, model, optimizer, \
    (avg_tr_loss, avg_val_loss, start_iter, start_epoch), \
    (tr_loss_results, val_loss_results, cer_results, wer_results) = _load_or_create_model(
        epochs, dataset, continue_from, learning_rate,
        rnn_type, hidden_size, hidden_layers,
        momentum, cuda, tensorboard_writer)

  (train_dataset, train_loader), (test_dataset, test_loader) = _get_datasets(dataset_dir, train_split, labels, batch, num_workers)

  best_wer = None
  batch_time, data_time, tr_losses, val_losses = _init_averages()

  print(model)
  print("Number of parameters: %d" % _model.LipReader.get_param_size(model))

  # josephz: CTCLoss, see https://github.com/SeanNaren/warp-ctc
  criterion = CTCLoss()
  decoder = _decoder.GreedyDecoder(labels)

  for epoch in range(start_epoch, epochs):
    model.train()
    epoch_start = time.time()

    for i, (data) in enumerate(train_loader, start=start_iter):
      batch_start = time.time()

      inputs, targets, input_percentages, target_sizes = data
      assert len(inputs.shape) == 4 and inputs.shape[2:] == (68, 3)
      batch_size, seq_len, num_pts, pts_dim = inputs.shape
      input_sizes = input_percentages.mul(int(inputs.size(1))).int()

      # Measure elapsed data loading time.
      data_time.update(time.time() - batch_start)

      if cuda:
        inputs = inputs.cuda()

      out, output_sizes = model(inputs, input_sizes)
      out = out.transpose(0, 1)  # TxNxH

      # acts: Tensor of (seqLength x batch x outputDim) containing output activations from network (before softmax)
      # labels: 1 dimensional Tensor containing all the targets of the batch in one large sequence
      # act_lens: Tensor of size (batch) containing size of each output sequence from the network
      # label_lens: Tensor of (batch) containing label length of each example
      assert len(targets.shape) == 1
      assert len(out.shape) == 3 and out.shape[:2] == (seq_len, batch_size)
      tr_loss = criterion(out, targets, output_sizes, target_sizes)
      # Average loss by minibatch.
      tr_loss /= inputs.size(0)

      val_loss_value = tr_loss.item()
      if val_loss_value == np.inf or val_loss_value == -np.inf:
        print("WARNING: received an inf loss, setting loss value to 0")
        val_loss_value = 0

      avg_tr_loss += val_loss_value
      tr_losses.update(val_loss_value, inputs.size(0))

      # Compute gradient.
      optimizer.zero_grad()
      tr_loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

      # SDG step!
      optimizer.step()

      # Measure elapsed batch time.
      batch_time.update(time.time() - batch_start)
      if not silent:
        print('Epoch[{}][{}{}]'.format(epoch + 1, i + 1, len(train_loader)), end='\t')
        print('Time {:0.3f} ({:0.3f})'.format(batch_time.val, batch_time.avg), end='\t')
        print('Data {:0.3f} ({:0.3f})'.format(data_time.val, data_time.avg), end='\t')
        print('Loss {:0.4f} ({:0.4f})'.format(tr_losses.val, tr_losses.avg))
    avg_tr_loss /= len(train_loader)

    print('Training Summary Epoch: [{}]'.format(epoch + 1), end='\t')
    print('Time taken (s): {:0.0f}'.format(time.time() - epoch_start))
    print('Time taken (s): {:0.0f}'.format(time.time() - epoch_start))
    print('Average Training Loss: {:0.3f}'.format(avg_tr_loss))

    # Reset start iteration in preparation for next epoch.
    start_iter = 0

    total_cer = total_wer = 0
    model.eval()
    with torch.no_grad():
      for i, (data) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(1))).int()
        batch_size, seq_len, num_pts, pts_dim = inputs.shape

        # Unflatten targets?
        split_targets = []
        offset = 0
        for size in target_sizes:
          split_targets.append(targets[offset:offset + size])
          offset += size

        if cuda:
          inputs = inputs.cuda()

        out, output_sizes = model(inputs, input_sizes)
        out_loss = out.transpose(0, 1)  # TxNxH
        assert len(targets.shape) == 1
        assert len(out_loss.shape) == 3 and out_loss.shape[:2] == (seq_len, batch_size)
        # out is supposed to be (seqLength x batch x outputDim).
        val_loss = criterion(out_loss, targets, output_sizes, target_sizes)

        val_loss_value = val_loss.item()
        if val_loss_value == np.inf or val_loss_value == -np.inf:
          print("WARNING: received an inf loss, setting loss value to 0")
          val_loss_value = 0

        avg_val_loss += val_loss_value
        val_losses.update(val_loss_value, inputs.size(0))

        decoded_output, _ = decoder.decode(out.data, output_sizes)
        target_strings = decoder.convert_to_strings(split_targets)

        for x in range(len(target_strings)):
          transcript, reference = decoded_output[x][0], target_strings[x][0]
          total_wer += decoder.wer(transcript, reference) / float(len(reference.split()))
          total_cer += decoder.cer(transcript, reference) / float(len(reference))
      avg_val_loss /= len(test_loader)

      val_loss_results[epoch] = avg_val_loss
      wer = wer_results[epoch] = 100 * total_wer / len(test_loader.dataset)  # .dataset?
      cer = cer_results[epoch] = 100 * total_cer / len(test_loader.dataset)

      print('Validation Summary Epoch: [{}]'.format(epoch + 1), end='\t')
      print('Average WER: {:0.3f}'.format(wer_results[epoch]), end='\t')
      print('Average CER: {:0.3f}'.format(cer_results[epoch]), end='\t')
      print('Average Validation Loss: {:0.3f}'.format(avg_val_loss))

      if tensorboard:
        _tensorboard_log(tensorboard_writer, dataset, epoch + 1, avg_val_loss, wer, cer, mode="Validation")
      if checkpoint:
        weights_path = _get_checkpoint_filepath(weights_dir, epoch + 1)
        torch.save(
          _model.LipReader.serialize(model, optimizer=optimizer,
            epoch=epoch, loss_results=val_loss_results, wer_results=wer_results,
            cer_results=cer_results), weights_path)

      # Do annealing.
      optim_state = optimizer.state_dict()
      optim_state['param_groups'][0]['lr'] /= annealing
      optimizer.load_state_dict(optim_state)

      if best_wer is None or best_wer > wer_results[epoch]:
        print('Found better validated model, saving to {}'.format)
        model_path = os.path.join(weights_dir, 'model.pth')
        weights_path = _get_checkpoint_filepath(weights_dir, epoch + 1)

        if os.path.isfile(weights_path):
          shutil.copyfile(weights_path, model_path)
        else:
          torch.save(
            _model.LipReader.serialize(model, optimizer=optimizer,
              epoch=epoch, loss_results=val_loss_results, wer_results=wer_results,
              cer_results=cer_results), model_path)
        best_wer = wer
        avg_tr_loss = 0

def main():
  global _logger
  args = _cmd.parseArgsForClassOrScript(train)
  varsArgs = vars(args)
  verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
  _getSharedLogger(verbosity=verbosity).info("Passed arguments: '{}'".format(varsArgs))
  train(**varsArgs)

if __name__ == '__main__':
  main()
