#!/usr/bin/env python3
"""
train.py
---

This trains a specified model.

"""
import os
import time

import numpy as np
import torch
import torch.utils.data as _data
import tensorboardX

import src.data.data_loader as _data_loader
import src.models.lipreader.better_model as _better_model
import src.train.train_better_model as _train
import src.utils.cmd_line as _cmd
import src.utils.utility as _util
import src.models.lipreader.analysis as _analysis

_logger = None

def _getSharedLogger(verbosity=_util.DEFAULT_VERBOSITY):
  global _logger
  if _logger is None:
   _logger = _util.getLogger(os.path.basename(__file__).split('.')[0], verbosity=verbosity)
  return _logger

def _get_datasets(dataset_name, train_split, sentence_dataset,
    threshold=0.8,
    labels='labels.json', rand=None, refresh=False, include_test=True):

  # REVIEW josephz: If we can load from pickles, we should not even do this split. We could have a helper factory thingy?
  # Load dataset video IDs and shuffle predictably.
  train_ids, val_ids, test_ids = _data_loader.split_dataset(dataset_name, train_split=train_split, rand=rand)

  train_dataset = _data_loader.FrameCaptionDataset(dataset_name, 'train', train_ids,
    labels=labels, threshold=threshold, sentence_dataset=sentence_dataset, refresh=refresh)
  val_dataset = _data_loader.FrameCaptionDataset(dataset_name, 'val', val_ids,
    labels=labels, threshold=threshold, sentence_dataset=sentence_dataset, refresh=refresh)
  if include_test:
    test_dataset = _data_loader.FrameCaptionDataset(dataset_name, 'test', test_ids,
      labels=labels, threshold=threshold, sentence_dataset=sentence_dataset, refresh=refresh)

  print()
  print("Dataset Information:")
  print("\tTrain Dataset Size:", len(train_dataset))
  print("\tVal Dataset Size:", len(val_dataset))
  if include_test:
    print("\tTest Dataset Size:", len(test_dataset))
  print()
  return (train_dataset, val_dataset, test_dataset) if include_test else (train_dataset, val_dataset)

def _init_models(
    char2idx,
    num_layers,
    frame_dim,
    hidden_size,
    char_dim,
    enable_ctc,

    rnn_type,
    attention_type,
    attn_hidden_size,
    bidirectional,
    rnn_dropout,
    device
):
  encoder = _better_model.VideoEncoder(frame_dim, hidden_size,
    rnn_type=rnn_type, num_layers=num_layers, bidirectional=bidirectional, rnn_dropout=rnn_dropout,
    enable_ctc=enable_ctc, vocab_size=len(char2idx), char2idx=char2idx, device=device).to(device)
  decoding_step = _better_model.CharDecodingStep(encoder,
    char_dim=char_dim, vocab_size=len(char2idx), char2idx=char2idx, rnn_dropout=rnn_dropout, attention_type=attention_type,
    attn_hidden_size=attn_hidden_size, device=device).to(device)

  return encoder, decoding_step


def restore(net, save_file):
  """Restores the weights from a saved file

  This does more than the simple Pytorch restore. It checks that the names
  of variables match, and if they don't doesn't throw a fit. It is similar
  to how Caffe acts. This is especially useful if you decide to change your
  network architecture but don't want to retrain from scratch.

  Args:
      net(torch.nn.Module): The net to restore
      save_file(str): The file path
  """

  net_state_dict = net.state_dict()
  restore_state_dict = torch.load(save_file)

  restored_var_names = set()

  print('\tRestoring:')
  for var_name in restore_state_dict.keys():
    if var_name in net_state_dict:
      var_size = net_state_dict[var_name].size()
      restore_size = restore_state_dict[var_name].size()
      if var_size != restore_size:
        print('\t\tShape mismatch for var', var_name, 'expected', var_size, 'got', restore_size)
      else:
        if isinstance(net_state_dict[var_name], torch.nn.Parameter):
          # backwards compatibility for serialized parameters
          net_state_dict[var_name] = restore_state_dict[var_name].data
        try:
          net_state_dict[var_name].copy_(restore_state_dict[var_name])
          print(str(var_name) + ' -> \t' + str(var_size) + ' = ' + str(int(np.prod(var_size) * 4 / 10**6)) + 'MB')
          restored_var_names.add(var_name)
        except Exception as ex:
          print('\t\tWhile copying the parameter named {}, whose dimensions in the model are'
                ' {} and whose dimensions in the checkpoint are {}, ...'.format(
            var_name, var_size, restore_size))
          raise ex

  ignored_var_names = sorted(list(set(restore_state_dict.keys()) - restored_var_names))
  unset_var_names = sorted(list(set(net_state_dict.keys()) - restored_var_names))
  if len(ignored_var_names) == 0:
    print('\t\tRestored all variables')
  else:
    print('\t\tDid not restore:\n\t' + '\n\t'.join(ignored_var_names))
  if len(unset_var_names) == 0:
    print('\t\tNo new variables')
  else:
    print('\t\tInitialized but did not modify:\n\t' + '\n\t'.join(unset_var_names))

  print('\tRestored %s' % save_file)

def train(
    data="StephenColbert/2",
    labels="labels.json",
    sentence_dataset=False,
    occlussion_threshold=0.8,
    train_split=0.8,
    num_workers=1,
    refresh=False,

    patience=10,
    batch_size=4,
    learning_rate=1e-4,
    weight_decay=1e-5,
    annealings=2,
    enable_ctc=False,
    grad_norm=50,

    tr_epochs=50,
    max_tfr=0.9,
    min_tfr=0.0,

    num_layers=1,
    frame_dim=68*3,
    hidden_size=700,
    char_dim=300,

    rnn_type='LSTM',
    attention_type='1_layer_nn',
    attn_hidden_size=-1,
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
  :param patience:
  :param batch_size:
  :param learning_rate:
  :param annealings: Number of times to anneal learning rate before training is finished.
  :param enable_ctc:
  :param max_tfr:
  :param grad_norm:
  :param num_layers:
  :param frame_dim:
  :param hidden_size:
  :param char_dim:
  :param rnn_type:
  :param attention_type:
  :param attn_hidden_size:
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
  print("Device: ", device)

  # Init Data.
  print("Initializing dataset '{}'".format(data))
  train_dataset, val_dataset, test_dataset = _get_datasets(data, train_split, sentence_dataset,
    threshold=occlussion_threshold, labels=labels, rand=rand, refresh=refresh, include_test=True)
  train_loader = _data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_data_loader._collate_fn)
  val_loader = _data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_data_loader._collate_fn)
  test_loader = _data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_data_loader._collate_fn)
  # Init Models.
  print("Initializing model")
  encoder, decoding_step = _init_models(train_dataset.char2idx, num_layers, frame_dim, hidden_size, char_dim,
    enable_ctc, rnn_type, attention_type, attn_hidden_size, bidirectional, rnn_dropout, device)

  # Initialize Logging.
  weights_dir = _util.getRelWeightsPath(data, use_existing=False)

  tensorboard_writer = tensorboardX.SummaryWriter(weights_dir)
  _getSharedLogger().info("Writing Tensorboard logs to '%s'", weights_dir)
  print()
  print("Try visualizing by running the following:")
  print(f"\ttensorboard --logdir='{weights_dir}'")
  print("Then open the following URL in your local browser. "
        "\n\tIf you're running on a remote machine see `README_TENSORBOARD.md` for help...")

  # REVIEW josephz: Multi-input support doesn't seem ready yet: https://github.com/lanpa/tensorboardX/issues/256
  # tensorboard_writer.add_graph(encoder,
  #   torch.autograd.Variable(
  #     torch.tensor([torch.zeros(batch_size, 100, 68, 3), torch.zeros(batch_size,))))
  # tensorboard_writer.add_graph(decoding_step,
  #   torch.autograd.Variable(
  #     torch.tensor(torch.zeros(batch_size,), torch.zeros(num_layers, batch_size, hidden_size), torch.zeros(batch_size,), torch.zeros(batch_size, 100,
  #       hidden_size))))

  # Train.
  val_cers = []
  train_decoder_losses = []
  train_ctc_losses = []

  best_val_cer = 1.0
  best_val_cer_idx = -1

  # Initial evaluation
  print("Initial evaluation...")
  decoder_loss, val_correct, val_count = _train.eval(encoder, decoding_step, val_loader, device, train_dataset.char2idx)
  val_cer = (val_count - val_correct).float() / val_count
  print("\tCER: ", str(val_cer))

  encoder_path = os.path.join(weights_dir, "best_encoder.pth")
  decoder_path = os.path.join(weights_dir, "best_decoder.pth")

  num_epochs = 0
  num_annealings = 0

  print("Beginning training loop")
  ts = time.time()
  while val_cer < best_val_cer or num_annealings < annealings:
    print("Epoch {}:".format(num_epochs + 1))

    if num_epochs - best_val_cer_idx > patience:
      # If the model does not improve after our set 'patience' number of epochs, we will reduce the learning rate.
      num_annealings += 1
      learning_rate /= 5
      print(f'\tAnnealing to {learning_rate}')
      restore(encoder, encoder_path)
      restore(decoding_step, decoder_path)

      # Must set best val CER to here, or else this will also trigger next loop
      # if val CER does not go down.
      best_val_cer_idx = num_epochs

    # Apply linear teacher-forcing ratio decay.
    curr_tfr = max(min_tfr, max_tfr - num_epochs / tr_epochs)
    assert 0.0 <= curr_tfr <= 1.0
    print(f'\tCurrent Teacher Forcing Ratio: {curr_tfr}')

    avg_decoder_loss, avg_ctc_loss = _train.train(encoder, decoding_step, train_loader,
      opt=torch.optim.Adam(list(encoder.parameters()) + list(decoding_step.parameters()), lr=learning_rate,
          weight_decay=weight_decay),
      device=device,
      char2idx=train_dataset.char2idx,
      teacher_forcing_ratio=curr_tfr,
      grad_norm=grad_norm)
    print(f'\tAVG Decoder Loss: {avg_decoder_loss}')
    print(f'\tAVG CTC Loss: {avg_ctc_loss}')
    tensorboard_writer.add_scalar(os.path.join(data, 'avg decoder loss'), avg_decoder_loss, global_step=num_epochs)
    tensorboard_writer.add_scalar(os.path.join(data, 'avg CTC loss'), avg_ctc_loss, global_step=num_epochs)

    decoder_loss, val_correct, val_count = _train.eval(encoder, decoding_step, val_loader, device, train_dataset.char2idx)
    _, train_correct, train_count = _train.eval(encoder, decoding_step, train_loader, device, train_dataset.char2idx)

    val_cer = (val_count - val_correct).float() / val_count
    train_cer = (train_count - train_correct).float() / train_count

    encoder.save_best_model(val_cer, encoder_path)
    decoding_step.save_best_model(val_cer, decoder_path)

    print(f'\tTrain CER: {train_cer}')
    print(f'\tVal CER: {val_cer}')

    if num_epochs % 10 == 0:
      print("\tTesting...")
      _train.test(data, encoder, decoding_step, test_loader, device, train_dataset, test_dataset, num_epochs)
    tensorboard_writer.add_scalars(os.path.join(data, 'CER'), {"Train": train_cer, "Val": val_cer}, global_step=num_epochs)
    tensorboard_writer.add_scalar(os.path.join(data, 'learning rate'), learning_rate, global_step=num_epochs)

    val_cers.append(val_cer)
    train_decoder_losses.append(avg_decoder_loss)
    train_ctc_losses.append(avg_ctc_loss)

    if val_cer < best_val_cer:
      best_val_cer = val_cer
      best_val_cer_idx = num_epochs

    num_epochs += 1

  # Finished training, final test.
  _train.test(data, encoder, decoding_step, test_loader, device, train_dataset, test_dataset, num_epochs)

  te = time.time()
  total_time = te - ts
  print()
  print("Training complete: Took '{}' seconds, or '{}' per epoch".format(total_time, total_time / num_epochs))
  print("Training Statistics")
  print("\tBest Val CER: '{}'".format(np.min(val_cers)))
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
