#!/usr/bin/env python3
"""
video.py
---

Video reading utilities.

"""

import os

import imageio
import collections

import numpy as np

class VideoReader:
  def __init__(self, vid_path, start=0, seq_len=1, fps=29.97):
    """ The VideoReader serves a generator of frame-sequences as needed. To increase performance, sequential
    frame access is enforced, and frames are cached into a buffer to allow for quicker repeated and
    sequential accesses within the allocated buffer.
    TODO: Buffer implementation. Could be implemented by taking a cache_size param for max(caption_size) and
    keeping track lo/hi indices for when to update. See `_updateCache` for draft.

    Note memory usage may grow significantly with high-resolution video or large cache sizes, calculated by the
    following formula:
    ```
    bytes = seq_len * height * width * channels
    ```

    A 1080p (1920x1080) video with sequence length of 30, or approximately 1 second of 30fps footage equates to:
    ```
    30 * 1080 * 1920 * 3 bytes ~ 186MB
    ```

    :param vid_path: Path of the video to read from.
    :param start: The starting frame index to begin reading from.
    :param seq_len: The length of the sequence of frames to serve.
    """
    assert os.path.isfile(vid_path)
    reader = imageio.get_reader(vid_path)
    vid_len = reader.get_length()

    # State.
    self.lo = start
    self._reader = reader
    self.cache = collections.deque(maxlen=seq_len)
    self.buf = np.empty(shape=(seq_len,), dtype=np.ndarray)

    # For convenience.
    self._seq_len = seq_len
    self._vid_len = vid_len
    self._fps = fps

  def get_frame_idx(self, seconds):
    return int(seconds * self._fps)

  def getNumFrames(self):
    return self._vid_len

  def genFrames(self, lo, hi):
    # assert isinstance(self.buf, np.ndarray) and self.buf.ndim == 1 and len(self.buf) == self._seq_len
    # assert isinstance(self.cache, collections.deque) and self.cache.maxlen == len(self.cache) == self._seq_len
    # self._updateCache(lo, hi)

    # Populate ndarray vector with cache.
    # REVIEW josephz: Can this be improved with a buffer?
    assert self.lo <= lo <= hi
    self.lo = lo
    return [self._reader.get_data(x) for x in range(lo, min(hi, self.getNumFrames()))]
    # for x in range(lo, hi):
    #   yield self._reader.get_data(x)

  def _updateCache(self, lo, hi):
    raise NotImplementedError
    # assert isinstance(self.cache, collections.deque) and self.cache.maxlen == self._seq_len
    # assert self.lo <= lo < hi < self._vid_len
    #
    # # The subsequent sequence may jump ahead. If so, we only wish to load the minimum number of
    # # frames to catch-up since our previous sequence.
    # assert lo + len(self.cache) <= hi + self._seq_len
    # cacheRange = range(max(lo + len(self.cache), hi), hi + self._seq_len)
    # self.cache.extend(self._reader.get_data(x) for x in cacheRange)
