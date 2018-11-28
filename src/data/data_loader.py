"""
data_loader.py
---

  Default Data Loader.

"""

import os
import numpy as np
import spacy

import torch
import torch.utils.data as _data
from scipy.io.idl import AttrDict

import src.utils.utility as _util

_logger = None

def _getSharedLogger(verbosity=_util.DEFAULT_VERBOSITY):
  global _logger
  if _logger is None:
    _logger = _util.getLogger(os.path.basename(__file__).split('.')[0], verbosity=verbosity)
  return _logger

BOS = '<BOS>'
EOS = '<EOS>'
PAD = '<PAD>'
UNK = '<UNK>'
_markers2Id = {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
_id2Markers = {v:k for k, v in _markers2Id.items()}

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
    # This will gather the dataview:
    #   (num_captions, [seq_len_i, num_pts, 3]) to (N, [num_captions_i, {seq_len_j, num_pts, 3}])
    # along with the associated captions (N, [num_captions_i, "str..."]),
    # for 'N' number of videos, and 'num_captions' for the length of each caption set.
    assert len(frame_paths) == len(caption_paths)
    dataviews = [np.load(x) for x in frame_paths]
    captions = [np.load(x) for x in caption_paths]
    dataview = np.concatenate(dataviews, axis=0)
    captions = np.concatenate(captions, axis=0)

    assert len(dataview.shape) == 1, "dataview.shape: '{}'".format(dataview.shape)
    assert all(len(x.shape) == 3 for x in dataview)
    assert len(captions.shape) == 1
    assert all(isinstance(x, str) for x in captions)
    assert len(dataview) == len(captions)
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
    """ Returns the frame_seq and caption at index.

    :param index:
    :return: Returns a tuple with the input data and caption:
      - frame_seq of shape [seq_len, num_pts, 3].
      - parsed_cap of shape [cap_len,].
    """

    frames = self.dataview[index]
    caption = self.captions[index]
    # parsed_cap = self.parse_caption(caption)

    frames = torch.FloatTensor(frames)
    # parsed_cap = torch.IntTensor(parsed_cap)

    assert len(frames.shape) == 3 and frames.shape[2] == 3
    # assert len(parsed_cap.shape) == 1
    return frames, caption

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
    res = torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
    return res

  # Sort the batch by seq_len.
  batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
  assert all(len(x) == 2 for x in batch)
  frames, caps = zip(*batch)
  # REVIEW josephz: Frames should be of size (batch_len=1, seq_len, 68, 3),
  # but are currently just (seq_len, 68, 3).
  # print('len(frames)', len(frames), 'len(caps)', len(caps))

  # Get max len for poadding. We will use this to zero-pad the ends.
  max_seqlength = frames[0].size(0)

  # Get minibatch size, make sure it's not 0.
  assert 0 < len(frames) == len(caps)
  minibatch_size = len(frames)

  # Accumulate batch here. Because we are dealing with variable seq_len data,
  # we need to also track the 'actual length', or the length of data that is
  # not 'padding'.
  inps = []
  targets = []
  input_percentages = torch.FloatTensor(minibatch_size)
  target_lengths = torch.IntTensor(minibatch_size)

  for i in range(minibatch_size):
    frame_seq = frames[i]
    caption = caps[i]
    assert isinstance(frame_seq, torch.Tensor)
    assert len(frame_seq.shape) == 3 and frame_seq.shape[1:] == (68, 3)

    # Pad input, and track its length.
    padded_inp = pad(frame_seq, max_seqlength, dim=0)
    inps.append(padded_inp)
    seq_length = len(padded_inp)
    input_percentages[i] = seq_length / float(max_seqlength)

    # Track caption lengths.
    targets.extend(caption)
    target_lengths[i] = len(caption)
  inps = torch.stack(inps, dim=0)
  targets = torch.IntTensor(targets)
  return inps, targets, input_percentages, target_lengths

class FrameCaptionSentenceDataset(_data.Dataset):
  def __init__(self, vid_id_dirs, labels, batch_size=4,
      cap='cap', frame_type='face_lmk_seq', ext='.npy'):
    super(FrameCaptionSentenceDataset, self).__init__()
    assert all(os.path.isdir(x) for x in vid_id_dirs)
    assert frame_type in ('face_lmk_seq', 'face_vtx_seq')

    frame_paths = tuple(os.path.join(x, frame_type + ext) for x in vid_id_dirs)
    caption_paths = tuple(os.path.join(x, cap + ext) for x in vid_id_dirs)
    assert all(os.path.isfile(x) for x in frame_paths)

    # REVIEW josephz: This will fail upon loading an exorbitant amount of memory.
    # If this cannot fit into memory, we will need to load fnames at `__getitem__` time.
    assert len(frame_paths) == len(caption_paths)
    dataviews = [np.load(x) for x in frame_paths]
    captions = [np.load(x) for x in caption_paths]
    assert len(dataviews) == len(captions)

    # Sentence-wise splitting.
    new_dataviews, new_captions = FrameCaptionSentenceDataset.split_sentences(dataviews, captions)

    # [N, (seq_len, 68, 3)]
    # REVIEW josephz: Sort by seq_len.
    dataviews = np.concatenate(new_dataviews, axis=0)
    captions = np.concatenate(new_captions, axis=0)
    assert len(captions) == len(dataviews)
    assert all(len(x.shape) == 3 for x in dataviews)
    assert all(isinstance(x, str) for x in captions)

    chunk_size = len(captions) // batch_size
    num_chunks = len(captions) // chunk_size
    num_batches = chunk_size
    num_minibatches = num_batches * num_chunks

    # Construct batches.
    batches = FrameCaptionSentenceDataset.construct_batches(dataviews, captions, num_batches, batch_size, chunk_size)

    # Construct vocabulary.
    char2idx = FrameCaptionSentenceDataset.build_vocab(labels)

    # Cache all rows.
    self.char2idx = char2idx
    self.idx2char = {v:k for k, v in char2idx.items()}
    self.dataviews = dataviews
    self.captions = captions
    self.batches = batches

    # Cache dataview paths.
    self.frame_type = frame_type
    self.vid_id_dirs = vid_id_dirs
    self.frame_paths = frame_paths

    self.batch_size = batch_size
    self.num_batches = num_batches

  @staticmethod
  def split_sentences(dataviews, captions):
    nlp = spacy.load('en')
    new_dataviews, new_captions = [], []
    for frames, caps in zip(dataviews, captions):
      new_frames, new_caps = [], []
      left = 0
      right = 1
      while left < len(caps) and right < len(caps):
        cap = " ".join(caps[left:right])
        doc = nlp(cap)
        sentences = [x.string.strip() for x in doc.sents]
        if len(sentences) >= 2:
          cap = " ".join(caps[left:right - 1])
          new_frames.append(np.concatenate(frames[left:right - 1]))
          new_caps.append(cap)
          left = right - 1
        right += 1
      new_dataviews.append(new_frames)
      new_captions.append(new_caps)
    return new_dataviews, new_captions

  @staticmethod
  def construct_batches(dataviews, captions, num_batches, batch_size, chunk_size):
    batches = []
    for i in range(num_batches):
      batch_frames = []
      batch_captions = []
      for b in range(batch_size):
        idx = int(b * chunk_size) + i
        batch_frame = dataviews[idx]
        batch_caption = captions[idx]
        batch_frames.append(batch_frame)
        batch_captions.append(batch_caption)
      batches.append((batch_frames, batch_captions))
    return batches

  @staticmethod
  def build_vocab(alphabet):
    char2idx = {}
    for k, v in _markers2Id.items():
      char2idx[k] = v
    for char in alphabet:
      char2idx[char] = len(char2idx)
    return char2idx

  def __len__(self):
    return self.num_batches

  def __getitem__(self, index):
    """ Returns the frame_seq and caption at index.

    :param index:
    :return: Returns a tuple with the input data and caption:
      - frame_seq of shape [batch_size, (seq_len, num_pts, 3)].
      - parsed_cap of shape [batch_size, (cap_len,)].
    """
    batch = self.batches[index]
    assert len(batch) == 2
    frames, captions = batch
    assert len(frames) == len(captions)
    assert all(len(x.shape) == 3 and x.shape[2] == 3 for x in frames)

    # Convert captions to indices and add BOS/EOS markers.
    parsed_captions = self.parse_captions(captions)
    parsed_captions = np.array(parsed_captions)
    return frames, parsed_captions

  def parse_captions(self, caps):
    # REVIEW josephz: According to the documentation filter has the following behavior:
    # "Note that `filter(function, iterable)` is equivalent to the generator expression
    # `(item for item in iterable if function(item))` if function is not `None` and
    # `(item for item in iterable if item)` if function is `None`."
    # That would imply that the below can be simplified to just
    # `[self.labels_map.get(x) for x in cap]`?
    # return list(filter(None, [self.labels_map.get(x) for x in list(cap)]))
    # Ah, it's actually different in that 'None' values are ignored.
    res = []
    for cap in caps:
      ids = [_markers2Id[BOS]]
      # Gets index of character for caption. UNK id if for whatever reason is not in dataset.
      ids.extend([self.char2idx.get(x, _markers2Id[UNK]) for x in list(cap)])
      ids.append(_markers2Id[EOS])
      ids = np.array(ids)
      assert len(ids) > 2 and ids[0] == _markers2Id[BOS] and ids[-1] == _markers2Id[EOS]
      res.append(ids)
    return res

def _collate_sentences_fn(batches):
  """ Collates batches sentence-wise. """
  # Sort the batch by seq_len.
  # batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)

  def _pad(seqs):
    """ Pads a batch of sequences of varying seq_len. """
    # pad_size = list(vec.shape)
    # pad_size[dim] = seq_len - vec.size(dim)
    # res = torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
    assert len(seqs) > 0 and all(x.shape[1:] == seqs[0].shape[1:] for x in seqs)
    lens = torch.LongTensor([len(x) for x in seqs])
    max_seq_len = torch.max(lens)

    # padded_seq_dims: (batch, max_seq_len, ...).
    padded_seq_dims = (len(seqs), max_seq_len,) + seqs[0].shape[1:]
    res = torch.zeros(padded_seq_dims)
    for i, seq in enumerate(seqs):
      src_len = lens[i]
      res[i, :src_len] = torch.LongTensor(seq)
    return res, lens

  assert all(len(x) == 2 for x in batches)
  # (1, batch, (seq_len, 68, 3))
  frames, captions = zip(*batches)

  # Merge sequences (from tuple of 1D tensor to 2D tensor)
  # (batch, seq_len, ...)
  src_seqs, src_lens = _pad(frames[0])
  tgt_seqs, tgt_lens = _pad(captions[0])

  return src_seqs, src_lens, tgt_seqs, tgt_lens

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

