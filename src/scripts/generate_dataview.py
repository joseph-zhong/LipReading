#!/usr/bin/env python3
"""
generate_dataview.py
---

This reads the available video and caption files available and generates the corresponding dataviews.

A dataview is a dense table of data input and label pairs.
For our purposes, we will generate a table for each video-caption pair as follows:


| idx  |  start, end  | landmark seq tensor   | caption text  |
| ---- | ------------ | --------------------- | ------------- |
| `i`  | `(s_i, e_i)` | `(frames, lmks, yx)`  | `"str...."`   |


"""
import os
import sys
import glob
import time
import collections

import numpy as np

import src.utils.data.video as _video
import src.utils.data.caption as _caption
import src.utils.data.face as _face

import src.utils.cmd_line as _cmd
import src.utils.utility as _util

_logger = None
def _getSharedLogger(verbosity=_util.DEFAULT_VERBOSITY):
  global _logger
  if _logger is None:
    _logger = _util.getLogger(os.path.basename(__file__).split('.')[0], verbosity=verbosity)
  return _logger


def generate_dataview(
    inp="data/raw/StephenColbert/",
    outp_dir="datasets",
    vid_ext=".mp4",
    cap_ext=".vtt",
    out_ext=".npy",
    timedelay=0,
    force=False,
):
  """ Generates dataviews for the given input file or directory.

  :param inp: Input video/caption filename (without extension), or directory of video of video/captions.
    Videos must have extensions of 'vid_ext' and captions 'cap_ext' with coinciding names.
  :param outp_dir: Output directory for the datasets.
  :param force: Flag to overwrite existing files.
  """
  # Glob corresponding video and caption files.
  vid_glob = os.path.join(inp, '*' + vid_ext)
  cap_glob = os.path.join(inp, '*' + cap_ext)
  vid_paths = sorted(glob.glob(vid_glob))
  cap_paths = sorted(glob.glob(cap_glob))
  assert len(vid_paths) == len(cap_paths) > 0

  ts = time.time()
  _getSharedLogger().info("Processing '%d' number of videos")
  for idx in range(len(vid_paths)):
    # Get inp_paths.
    vid_path = vid_paths[idx]
    cap_path = cap_paths[idx]

    # Get dst_path.
    vid_basename = os.path.basename(vid_path).split('.')[0]
    cap_basename = os.path.basename(cap_path).split('.')[0]
    assert vid_basename == cap_basename

    _getSharedLogger().info("\tJob (%4d/%4d): Extracting captions and landmarks for '%s'",
        idx, len(vid_paths)-1, vid_path)

    dst_path = os.path.join(outp_dir, vid_basename + out_ext)
    # If force mode, or file doesn't exist, process job, else continue.
    if force or not os.path.isfile(dst_path):
      if force and os.path.isfile(dst_path):
        _getSharedLogger().warning("FORCE MODE ENABLED!!!! Overwriting existing file: '%s'...", dst_path)
      _getSharedLogger().info("\tJob (%4d/%4d): Writing dataview to '%s'",
        idx, len(vid_paths) - 1, dst_path)

      # Extract, prune, and filter captions.
      captions = _caption.extract_captions(cap_path)
      captions = _caption.prune_and_filter_captions(captions)

      # Extract face-dots.
      dataview = _get_captioned_landmarks(vid_path, captions, timedelay=timedelay)
      if force or not os.path.isfile(dst_path):
        _getSharedLogger().info("\tJob (%4d/%4d): Writing dataview to '%s'",
          idx, len(vid_paths) - 1, dst_path)
        np.save(dst_path, dataview)

  te = time.time()
  _getSharedLogger().info("\tJob (%4d/%4d): Done writing dataviews! Took %0.3f seconds",
    idx, len(vid_paths) - 1, te - ts)

# REVIEW josephz: How do I implement timedelay for this scenario? In the original implementation, the input
# audio signal was "delayed" by some ms, or the first `k` ms of input were cut along with the last corresponding
# `k` ms output. It is possible here that the opposite is true, where looking ahead in the mouth-shape will
# inform the model of the context? Or is even relevant? Or do we actually want to add frames on each side?
def _get_captioned_landmarks(vid_path, captions, timedelay=0):
  """ Extracts landmarks that coincide with the captions. Returns a dataview for a particular video-caption pair
  for frames that correspond with valid captions that also coincide a reasonably detectable face.
  """
  assert os.path.isfile(vid_path)
  assert isinstance(captions, collections.OrderedDict) and len(captions) > 0

  dataview = []
  video_reader = _video.VideoReader(vid_path)

  for (start, end), cap in captions.items():
    # REVIEW josephz: How to apply timedelay here?
    start_frame = video_reader.get_frame_idx(start) - timedelay
    end_frame = video_reader.get_frame_idx(end) + timedelay
    _getSharedLogger().info("Computing landmarks for '%d' frames", end_frame - start_frame)
    frames = video_reader.genFrames(start_frame, end_frame)

    ts = time.time()
    for i, frame in enumerate(frames):
      assert len(frame.shape) == 3
      lmks = _face.get_3d_landmarks(frame)
      # REVIEW josephz: How else to check that most of the landmarks are valid? Could also check the size of the
      #  face to be greater than a certain box.
      if lmks is not None:
        data = np.array([(start, end), lmks, cap])
        dataview.append(data)
        _getSharedLogger().info("\tJob (%4d/%4d): Generated data example for caption (start=%d, end=%d), "
                                "took '%0.3f' seconds",
          i, video_reader.getNumFrames() - 1, start, end, time.time() - ts)

  return dataview

def main(args):
  global _logger
  args = _cmd.parseArgsForClassOrScript(generate_dataview)
  varsArgs = vars(args)
  verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
  _getSharedLogger(verbosity=verbosity).info("Passed arguments: '{}'".format(varsArgs))
  generate_dataview(**varsArgs)

if __name__ == '__main__':
  main(sys.argv)