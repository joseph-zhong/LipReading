"""
data_loader.py
---

  Default Data Loader.

"""

import os
import numpy as np
import torch

import torch.utils.data as _data

import src.utils.utility as _util

_logger = None

def _getSharedLogger(verbosity=_util.DEFAULT_VERBOSITY):
  global _logger
  if _logger is None:
    _logger = _util.getLogger(os.path.basename(__file__).split('.')[0], verbosity=verbosity)
  return _logger

class FrameCaptionDataset(_data.Dataset):
  def __init__(self, vid_id_dirs, labels,
      cap='cap', frame_type='face_lmk_seq', ext='.npy'):
    """ Dataset that loads the video dataview and captions.

    :param vid_id_dirs: Directories of video ids to include in the dataset.
    :param labels: Corresponding dataset vocabulary.
    :param cap: Base filename for caption rows to load.
    :param frame_type: Frame type to use for input, also the base filename for frame rows to load.
    """
    super(FrameCaptionDataset, self).__init__()
    assert all(os.path.isdir(x) for x in vid_id_dirs)
    assert frame_type in ('face_lmk_seq', 'face_vtx_seq')

    frame_paths = tuple(os.path.join(x, frame_type + ext) for x in vid_id_dirs)
    caption_paths = tuple(os.path.join(x, cap + ext) for x in vid_id_dirs)
    assert all(os.path.isfile(x) for x in frame_paths)

    # REVIEW josephz: This will fail upon loading an exorbitant amount of memory.
    # If this cannot fit into memory, we will need to load fnames at `__getitem__` time.
    # try:
    # Gather lmk_seq/caption batch pairs across videos.
    dataview = np.concatenate([np.load(fname) for fname in frame_paths], axis=0)
    captions = np.concatenate([np.load(fname) for fname in caption_paths], axis=0)
    # except:
    #   _getSharedLogger().warning(
    #     "Failed to load dataview, DataSet will read data from disk instead...")
    #   raise NotImplementedError()


    # Cache all rows.
    self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
    self.dataview = dataview
    self.captions = captions

    # Cache dataview paths.
    self.frame_type = frame_type
    self.vid_id_dirs = vid_id_dirs
    self.frame_paths = frame_paths

  def __getitem__(self, index):
    frames = self.dataview[index]
    caption = self.captions[index]
    parsed_cap = self.parse_caption(caption)

    frames = torch.FloatTensor(frames)
    parsed_cap = torch.IntTensor(parsed_cap)
    return frames, parsed_cap

  def __len__(self):
    return len(self.dataview)

  def parse_caption(self, cap):
    # REVIEW josephz: According to the documentation filter has the following behavior:
    # "Note that `filter(function, iterable)` is equivalent to the generator expression
    # `(item for item in iterable if function(item))` if function is not `None` and
    # `(item for item in iterable if item)` if function is `None`."
    # That would imply that the below can be simplified to just
    # `[self.labels_map.get(x) for x in cap]`?
    # return list(filter(None, [self.labels_map.get(x) for x in list(cap)]))
    # Ah, it's actually different in that 'None' values are ignored.
    return list(filter(None, [self.labels_map.get(x) for x in list(cap)]))

# REVIEW josephz: What the fuck does this do????
# It seems in `dataloader.py` the following is used for
# `samples = collate_fn([dataset[i] for i in batch_indices])`.
# Now a question is does this actually enforce ordering?
def _collate_fn(batch):
  """
  Custom collate function to manually specify how samples are batched from the DataLoader.
  See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#iterating-through-the-dataset
  for an introduction.

  In particular, the batch represents a collection of
    (lmk_seq, caption)
  pairs. However each sequence may be of different sequence length.

  :param batch: The complete batch of data samples to be collated. In particular, this is a list of frames and captions,
    each of shape (seq_len, num_pts, pt_dim), (num_chars)
  """
  # Pad based on the maximum seq_len.
  def pad(vec, seq_len, dim=0):
    """
    :param vec: Tensor to pad.
    :param seq_len: Fixed target size..
    :param dim: Along which dimension to apply padding."""

    pad_size = list(vec.shape)
    pad_size[dim] = seq_len - vec.size(dim)
    print("pad: pad_size:", pad_size)
    res = torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
    return res

  # Sort the batch by seq_len.
  batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
  assert all(len(x) == 2 for x in batch)
  frames, caps = zip(*batch)

  max_seqlength = frames[0].size(0)
  lengths = [len(x) for x in caps]

  inps = []
  targets = torch.zeros(len(caps), max(lengths)).long()
  assert 0 < len(frames) == len(caps)
  for i, (x, y) in enumerate(zip(frames, caps)):
    # Pad input.
    padded_inp = pad(x, max_seqlength, dim=0)
    inps.append(padded_inp)

    # Pad label.
    targets[i, :len(y)] = y
  inps = torch.stack(inps, dim=0)
  return inps, targets

# REVIEW josephz: Is this also over-kill?
class BucketingSampler(_data.Sampler):
  def __init__(self, data_source, batch_size=1):
    """
    Samples batches assuming they are in order of size to batch similarly sized samples together.
    """
    super(BucketingSampler, self).__init__(data_source)
    self.data_source = data_source
    ids = list(range(0, len(data_source)))
    self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

  def __iter__(self):
    for ids in self.bins:
      np.random.shuffle(ids)
      yield ids

  def __len__(self):
    return len(self.bins)

  def shuffle(self, epoch):
    np.random.shuffle(self.bins)

