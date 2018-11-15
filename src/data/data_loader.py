"""
data_loader.py
---

  Default Data Loader.

"""

import os
import numpy as np

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
    assert all(os.path.isdir(x) for x in vid_id_dirs)
    assert frame_type in ('face_frames', 'face_lmk_seq', 'face_vtx_seq')

    frame_paths = (os.path.join(x, frame_type + ext) for x in vid_id_dirs)
    caption_paths = (os.path.join(x, cap + ext) for x in vid_id_dirs)
    assert all(os.path.isfile(x) for x in frame_paths)

    # REVIEW josephz: This will fail upon loading an exorbitant amount of memory.
    # If this cannot fit into memory, we will need to load fnames at `__getitem__` time.
    try:
      dataview = np.concatenate([np.load(fname) for fname in frame_paths], axis=0)
      captions = np.concatenate([np.load(fname) for fname in caption_paths], axis=0)
    except:
      _getSharedLogger().warning(
        "Failed to load dataview, DataSet will read data from disk instead...")
      raise NotImplementedError()

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

    return frames, parsed_cap

  def parse_caption(self, cap):
    # REVIEW josephz: According to the documentation filter has the following behavior:
    # "Note that `filter(function, iterable)` is equivalent to the generator expression
    # `(item for item in iterable if function(item))` if function is not `None` and
    # `(item for item in iterable if item)` if function is `None`."
    # That would imply that the below can be simplified to just
    # `[self.labels_map.get(x) for x in cap]`?
    return list(filter(None, [self.labels_map.get(x) for x in list(cap)]))

  def __len__(self):
    return len(self.dataview)

class SpectrogramDataset(_data.Dataset):
  def __init__(self, audio_conf, manifest_filepath, labels, normalize=False, augment=False):
    """
    Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
    a comma. Each new line is a different sample. Example below:

    /path/to/audio.wav,/path/to/audio.txt
    ...

    :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
    :param manifest_filepath: Path to manifest csv as describe above
    :param labels: String containing all the possible characters to map to
    :param normalize: Apply standard mean and deviation normalization to audio tensor
    :param augment(default False):  Apply random tempo and gain perturbations
    """
    with open(manifest_filepath) as f:
      ids = f.readlines()
    ids = [x.strip().split(',') for x in ids]
    self.ids = ids
    self.size = len(ids)
    self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
    super(SpectrogramDataset, self).__init__(audio_conf, normalize, augment)

  def __getitem__(self, index):
    sample = self.ids[index]
    audio_path, transcript_path = sample[0], sample[1]
    spect = self.parse_audio(audio_path)
    transcript = self.parse_transcript(transcript_path)
    return spect, transcript

  def parse_transcript(self, transcript_path):
    with open(transcript_path, 'r', encoding='utf8') as transcript_file:
      transcript = transcript_file.read().replace('\n', '')
    transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
    return transcript

  def __len__(self):
    return self.size

