"""
data_loader.py
---

  Default Data Loader.

"""

import os
import json
import glob
import pickle
import numpy as np

import spacy
import torch
import torch.utils.data as _data

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
_labels = [" ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", ">", "?", "@", "[", "]", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

def gen_vid_ids(dataset_name, rand=None):
  # Load videos and captions.
  dataset_dir = _util.getRelDatasetsPath(dataset_name)
  vid_ids = glob.glob(os.path.join(dataset_dir, '*'))
  assert len(vid_ids) > 0, f"No video ids found: '{dataset_dir}'"
  vid_ids.sort()
  if rand is not None:
    rand.shuffle(vid_ids)
  else:
    np.random.shuffle(vid_ids)
  return vid_ids

def split_dataset(dataset_name, train_split=0.8, rand=None):
  vid_ids = gen_vid_ids(dataset_name, rand=rand)

  # Split dataset into train, val, and testing.
  train_idx = int(train_split * len(vid_ids))
  val_test_size = len(vid_ids) - train_idx
  assert int(train_idx + val_test_size) == len(vid_ids)
  val_idx = train_idx + val_test_size // 2

  train_vid_ids = vid_ids[:train_idx]
  val_vid_ids = vid_ids[train_idx:val_idx]
  test_vid_ids = vid_ids[val_idx:]

  return train_vid_ids, val_vid_ids, test_vid_ids

def load_dataset(pickle_dir, out_ext='.pkl'):
  char2idx_path = os.path.join(pickle_dir, 'char2idx' + out_ext)
  frames_path = os.path.join(pickle_dir, 'frames' + out_ext)
  captions_path = os.path.join(pickle_dir, 'captions' + out_ext)
  assert os.path.isfile(char2idx_path), "File not found: '{}'".format(char2idx_path)
  assert os.path.isfile(frames_path), "File not found: '{}'".format(frames_path)
  assert os.path.isfile(captions_path), "File not found: '{}'".format(captions_path)

  with open(char2idx_path, 'rb') as fin:
    char2idx = pickle.load(fin)
  with open(frames_path, 'rb') as fin:
    frames = pickle.load(fin)
  with open(captions_path, 'rb') as fin:
    captions = pickle.load(fin)
  return char2idx, frames, captions

def filter_occlusions(frames, captions, start_ends, fps=29.97, threshold=0.8):
  filtered_frames = []
  filtered_captions = []
  for f, c, s_e in zip(frames, captions, start_ends):
    start, end = s_e
    # Filter out frame/caption samples where fewer than 80% of the frames of the total time-window were captured,
    # as well as requiring that the frame seqlen exceeds the length of the caption by at least two, to account for
    # later added BOS/EOS tokens.
    if (end - start) * fps * threshold <= len(f) and len(c) + 2 < len(f):
      filtered_frames.append(f)
      filtered_captions.append(c)
  return filtered_frames, filtered_captions

def sort_by_seqlen(frames, captions):
  frame_lens = [x.shape[0] for x in frames]
  indices = np.argsort(frame_lens)
  sorted_frames = np.array(frames)[indices]
  sorted_captions = np.array(captions)[indices]
  return sorted_frames, sorted_captions

def build_vocab(dataset_name, labels):
  raw_dir = _util.getRelRawPath(dataset_name)
  labels_path = os.path.join(raw_dir, labels)
  try:
    with open(labels_path) as label_file:
      labels = str(''.join(json.load(label_file)))
  except:
    labels = _labels
    _getSharedLogger().warning("Could not open '%s'... \n\tUsing hardcoded labels: '%s'", labels_path, labels)

  char2idx = {}
  for k, v in _markers2Id.items():
    char2idx[k] = v
  for char in labels:
    char2idx[char] = len(char2idx)
  return char2idx

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
  def _pad(seqs, dtype=torch.float32):
    """ Pads a batch of sequences of varying seq_len. """
    assert len(seqs) > 0 and all(x.shape[1:] == seqs[0].shape[1:] for x in seqs)
    lens = torch.LongTensor([len(x) for x in seqs])
    max_seq_len = torch.max(lens)

    # padded_seq_dims: (batch, max_seq_len, ...).
    padded_seq_dims = (len(seqs), max_seq_len,) + seqs[0].shape[1:]
    res = torch.zeros(padded_seq_dims, dtype=dtype)
    for i, seq in enumerate(seqs):
      src_len = lens[i]
      res[i, :src_len] = torch.Tensor(seq)
    return res, lens

  assert all(len(x) == 2 for x in batch)
  # (1, batch, (seq_len, 68, 3))
  frames, captions = zip(*batch)

  # Merge sequences (from tuple of 1D tensor to 2D tensor)
  # (batch, seq_len, ...)
  src_seqs, src_lens = _pad(frames, dtype=torch.float32)
  tgt_seqs, tgt_lens = _pad(captions, dtype=torch.long)
  return src_seqs, src_lens, tgt_seqs, tgt_lens

class FrameCaptionDataset(_data.Dataset):
  def __init__(self, dataset_name, split_name, vid_ids,
      labels='labels.json', start_end='s_e', threshold=0.8, fps=29.97,
      cap='cap', frame_type='face_lmk_seq', sentence_dataset=False,
      in_ext='.npy', out_ext='.pkl',
      refresh=False):
    """ Dataset that loads the video dataview and captions.

    :param vid_ids: Directories of video ids to include in the dataset.
    :param labels: Corresponding dataset vocabulary.
    :param cap: Base filename for caption rows to load.
    :param frame_type: Frame type to use for input, also the base filename for frame rows to load.
    """
    super(FrameCaptionDataset, self).__init__()
    assert all(os.path.isdir(x) for x in vid_ids)
    assert frame_type in ('face_lmk_seq', 'face_vtx_seq')

    pickle_dir = _util.getRelPicklesPath(dataset_name, 'sentence' if sentence_dataset else 'non-sentence', split_name)
    if refresh or not os.path.isdir(pickle_dir):
      char2idx, frames, captions = FrameCaptionDataset.construct_dataset(dataset_name, pickle_dir, vid_ids,
        labels=labels, start_end=start_end, cap=cap, frame_type=frame_type, sentence_dataset=sentence_dataset,
        in_ext=in_ext, fps=fps, threshold=threshold)
    else:
      char2idx, frames, captions = load_dataset(pickle_dir, out_ext=out_ext)
    assert len(frames) == len(captions) > 0

    # Cache all rows.
    self.char2idx = char2idx
    self.idx2char = {v: k for k, v in char2idx.items()}

    self.frames = frames
    self.captions = captions
    self.num_elements = len(captions)

    # Cache dataview paths.
    self.frame_type = frame_type

  def __getitem__(self, index):
    """ Returns the frame_seq and caption at index.

    :param index:
    :return: Returns a tuple with the input data and caption:
      - frame_seq of shape [seq_len, num_pts, 3].
      - parsed_cap of shape [cap_len,].
    """

    frames = self.frames[index]
    caption = self.captions[index]
    assert len(frames.shape) == 3

    # Convert captions to indices and add BOS/EOS markers.
    parsed_caption = self.parse_caption(caption)
    parsed_caption = np.array(parsed_caption)
    return frames, parsed_caption

  def __len__(self):
    return self.num_elements

  @staticmethod
  def construct_dataset(dataset_name, pickle_dir, vid_ids,
      labels='labels.json',
      start_end='s_e', cap='cap', frame_type='face_lmk_seq', sentence_dataset=False,
      in_ext='.npy', out_ext='.pkl',
      fps=29.97, threshold=0.8):
    frame_paths = tuple(os.path.join(x, frame_type + in_ext) for x in vid_ids)
    caption_paths = tuple(os.path.join(x, cap + in_ext) for x in vid_ids)
    start_ends_paths = tuple(os.path.join(x, start_end + in_ext) for x in vid_ids)
    assert all(os.path.isfile(x) for x in frame_paths)

    assert len(frame_paths) == len(caption_paths)
    frames = [np.load(x) for x in frame_paths]
    captions = [np.load(x) for x in caption_paths]
    start_ends = [np.load(x) for x in start_ends_paths]
    assert len(frames) == len(captions)

    # Sentence-wise splitting.
    if sentence_dataset:
      frames, captions = FrameCaptionDataset.split_sentences(frames, captions)

    # [N, (seq_len, 68, 3)]
    frames = np.concatenate(frames, axis=0)
    captions = np.concatenate(captions, axis=0)
    start_ends = np.concatenate(start_ends, axis=0)
    assert len(captions) == len(frames)
    assert all(len(x.shape) == 3 for x in frames)
    assert all(isinstance(x, str) for x in captions)

    # Filter captions where frames are mostly occluded or mis-detected.
    frames, captions = filter_occlusions(frames, captions, start_ends, fps=fps, threshold=threshold)
    # Sort captions by sequence length.
    frames, captions = sort_by_seqlen(frames, captions)

    # Construct vocabulary.
    char2idx = build_vocab(dataset_name, labels)

    # Pickle dataset contents.
    _util.mkdirP(pickle_dir)
    with open(os.path.join(pickle_dir, 'char2idx' + out_ext), 'wb') as fout:
      pickle.dump(char2idx, fout)
    with open(os.path.join(pickle_dir, 'frames' + out_ext), 'wb') as fout:
      pickle.dump(frames, fout)
    with open(os.path.join(pickle_dir, 'captions' + out_ext), 'wb') as fout:
      pickle.dump(captions, fout)
    return char2idx, frames, captions

  @staticmethod
  def split_sentences(dataviews, captions):
    nlp = spacy.load('en')
    new_frames, new_captions = [], []
    for frames, caps in zip(dataviews, captions):
      new_fs, new_caps = [], []
      left = 0
      right = 1
      while left < len(caps) and right < len(caps):
        cap = " ".join(caps[left:right])
        doc = nlp(cap)
        sentences = [x.string.strip() for x in doc.sents]
        if len(sentences) >= 2 and right - 1 - left > 0:
          cap = " ".join(caps[left:right - 1])
          new_fs.append(np.concatenate(frames[left:right - 1]))
          print("sentence:", cap)
          new_caps.append(cap)
          left = right - 1
        right += 1
      new_frames.append(new_fs)
      new_captions.append(new_caps)
    return new_frames, new_captions

  # REVIEW josephz: This is a copy of `FrameCaptionDataset.parse_caption`.
  def parse_caption(self, cap):
    ids = [_markers2Id[BOS]]
    # Gets index of character for caption. UNK id if for whatever reason is not in dataset.
    ids.extend([self.char2idx.get(x, _markers2Id[UNK]) for x in list(cap)])
    ids.append(_markers2Id[EOS])
    ids = np.array(ids)
    assert len(ids) > 2 and ids[0] == _markers2Id[BOS] and ids[-1] == _markers2Id[EOS]
    return ids