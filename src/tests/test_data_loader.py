#!/usr/bin/env python3
"""

"""



import src.utils.utility as _util

import src.scripts.train_model as _train

dataset="StephenColbert/test_dataset_loader"
batch=10
num_workers=1
hidden_size=800
hidden_layers=5
train_split=0.5
rnn_type="gru"
epochs=70
cuda=True
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

tensorboard_writer = _train._get_tensorboard_writer(weights_dir, tensorboard)

labels, model, optimizer, \
  (avg_loss, start_iter, start_epoch), \
  (loss_results, cer_results, wer_results) = _train._load_or_create_model(
    epochs, dataset, continue_from, learning_rate, rnn_type, hidden_size, hidden_layers, momentum, cuda, tensorboard_writer)

(train_dataset, train_loader), (test_dataset, test_loader) = _train._get_datasets(
  dataset_dir, train_split, labels, batch, num_workers)

for i, (data) in enumerate(train_loader, start=start_iter):
  assert len(data) == 2
  inp, label = data
  import pdb; pdb.set_trace()

  print("[i='{}'] [inp.shape='{}'] [type(label)='{}']".format(i, inp.shape, type(label)))



