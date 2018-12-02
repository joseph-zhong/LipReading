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
      start_end='s_e', threshold=0.8, fps=29.97,
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
    start_end_paths = tuple(os.path.join(x, start_end + ext) for x in vid_id_dirs)
    assert all(os.path.isfile(x) for x in frame_paths)
    assert all(os.path.isfile(x) for x in caption_paths)
    assert all(os.path.isfile(x) for x in start_end_paths)

    # REVIEW josephz: This will fail upon loading an exorbitant amount of memory.
    # If this cannot fit into memory, we will need to load fnames at `__getitem__` time.
    # try:
    # Gather lmk_seq/caption batch pairs across videos.
    # This will gather the dataview:
    #   (num_captions, [seq_len_i, num_pts, 3]) to (N, [num_captions_i, {seq_len_j, num_pts, 3}])
    # along with the associated captions (N, [num_captions_i, "str..."]),
    # for 'N' number of videos, and 'num_captions' for the length of each caption set.
    assert len(frame_paths) == len(caption_paths) == len(start_end_paths)
    frames = [np.load(x) for x in frame_paths]
    captions = [np.load(x) for x in caption_paths]
    start_ends = [np.load(x) for x in start_end_paths]
    assert len(frames) == len(captions)

    frames = np.concatenate(frames, axis=0)
    captions = np.concatenate(captions, axis=0)
    start_ends = np.concatenate(start_ends, axis=0)
    assert all(len(x.shape) == 3 for x in frames)
    assert all(isinstance(x, str) for x in captions)

    # REVIEW JOSEPHZ: filter out frames which are mostly occluded.
    frames, captions = FrameCaptionDataset.filter_occlusions(frames, captions, start_ends, fps=fps, threshold=threshold)
    frames, captions = FrameCaptionDataset.sort_by_seqlen(frames, captions)

    num_elements = len(captions)

    # Construct batches.
    # batches = FrameCaptionSentenceDataset.construct_batches(frames, captions, num_batches, batch_size, chunk_size)
    # self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
    # Construct vocabulary.
    # REVIEW josephz: This should be the static class of this object.
    char2idx = FrameCaptionSentenceDataset.build_vocab(labels)

    # Cache all rows.
    self.char2idx = char2idx
    self.idx2char = {v: k for k, v in char2idx.items()}

    self.frames = frames
    self.captions = captions
    self.num_elements = num_elements

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

    frames = self.frames[index]
    caption = self.captions[index]
    assert len(frames.shape) == 3

    # Convert captions to indices and add BOS/EOS markers.
    parsed_caption = self.parse_caption(caption)
    parsed_caption = np.array(parsed_caption)
    return frames, parsed_caption

  def __len__(self):
    return self.num_elements

  def parse_caption(self, cap):
    ids = [_markers2Id[BOS]]
    # Gets index of character for caption. UNK id if for whatever reason is not in dataset.
    ids.extend([self.char2idx.get(x, _markers2Id[UNK]) for x in list(cap)])
    ids.append(_markers2Id[EOS])
    ids = np.array(ids)
    assert len(ids) > 2 and ids[0] == _markers2Id[BOS] and ids[-1] == _markers2Id[EOS]
    return ids

  @staticmethod
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

  @staticmethod
  def sort_by_seqlen(frames, captions):
    frame_lens = [x.shape[0] for x in frames]
    indices = np.argsort(frame_lens)
    sorted_frames = np.array(frames)[indices]
    sorted_captions = np.array(captions)[indices]
    return sorted_frames, sorted_captions

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

class FrameCaptionSentenceDataset(_data.Dataset):
  def __init__(self, vid_id_dirs, labels,
      start_end='s_e',
      cap='cap', frame_type='face_lmk_seq', ext='.npy',
      fps=29.97, threshold=0.8):
    super(FrameCaptionSentenceDataset, self).__init__()

    frame_paths = tuple(os.path.join(x, frame_type + ext) for x in vid_id_dirs)
    caption_paths = tuple(os.path.join(x, cap + ext) for x in vid_id_dirs)
    start_ends_paths = tuple(os.path.join(x, start_end + ext) for x in vid_id_dirs)
    assert all(os.path.isfile(x) for x in frame_paths)

    # REVIEW josephz: This will fail upon loading an exorbitant amount of memory.
    # If this cannot fit into memory, we will need to load fnames at `__getitem__` time.
    assert len(frame_paths) == len(caption_paths)
    frames = [np.load(x) for x in frame_paths]
    captions = [np.load(x) for x in caption_paths]
    start_ends = [np.load(x) for x in start_ends_paths]
    assert len(frames) == len(captions)

    # REVIEW josephz: Sort by seq_len.
    # todo: Sort data examples by sequence length.
    # Sentence-wise splitting.
    frames, captions = FrameCaptionSentenceDataset.split_sentences(frames, captions)

    # [N, (seq_len, 68, 3)]
    # REVIEW JOSEPHZ: filter out frames which are mostly occluded.
    frames = np.concatenate(frames, axis=0)
    captions = np.concatenate(captions, axis=0)
    start_ends = np.concatenate(start_ends, axis=0)
    assert len(captions) == len(frames)
    assert all(len(x.shape) == 3 for x in frames)
    assert all(isinstance(x, str) for x in captions)

    frames, captions = FrameCaptionDataset.filter_occlusions(frames, captions, start_ends, fps=fps, threshold=threshold)
    frames, captions = FrameCaptionDataset.sort_by_seqlen(frames, captions)
    num_elements = len(captions)

    # Construct vocabulary.
    char2idx = FrameCaptionSentenceDataset.build_vocab(labels)

    # Cache all rows.
    self.char2idx = char2idx
    self.idx2char = {v:k for k, v in char2idx.items()}
    self.frames = frames
    self.captions = captions

    # Cache dataview paths.
    self.frame_type = frame_type
    self.num_elements = num_elements

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
          print("sentence:", cap)
          new_caps.append(cap)
          left = right - 1
        right += 1
      new_dataviews.append(new_frames)
      new_captions.append(new_caps)
    return new_dataviews, new_captions

  @staticmethod
  def build_vocab(alphabet):
    char2idx = {}
    for k, v in _markers2Id.items():
      char2idx[k] = v
    for char in alphabet:
      char2idx[char] = len(char2idx)
    return char2idx

  def __len__(self):
    return self.num_elements

  def __getitem__(self, index):
    """ Returns the frame_seq and caption at index.

    :param index:
    :return: Returns a tuple with the input data and caption:
      - frame_seq of shape [batch_size, (seq_len, num_pts, 3)].
      - parsed_cap of shape [batch_size, (cap_len,)].
    """
    frame = self.frames[index]
    caption = self.captions[index]
    assert len(frame.shape) == 3

    # Convert caption to indices and add BOS/EOS markers.
    parsed_caption = self.parse_caption(caption)
    parsed_caption = np.array(parsed_caption)
    return frame, parsed_caption

  def parse_caption(self, cap):
    ids = [_markers2Id[BOS]]
    # Gets index of character for caption. UNK id if for whatever reason is not in dataset.
    ids.extend([self.char2idx.get(x, _markers2Id[UNK]) for x in list(cap)])
    ids.append(_markers2Id[EOS])
    ids = np.array(ids)
    assert len(ids) > 2 and ids[0] == _markers2Id[BOS] and ids[-1] == _markers2Id[EOS]
    return ids