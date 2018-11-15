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
