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

import glob
import os
import sys

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
    inp="data/raw/StephenColbert",
    outp_dir="datasets",
    vid_ext=".mp4",
    cap_ext=".vtt",
    out_ext=".npy",
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
  assert len(vid_paths) == len(cap_paths)

  for idx in range(len(vid_paths)):
    # Get inp_paths.
    vid_path = vid_paths[idx]
    cap_path = cap_paths[idx]

    # Get dst_path.
    vid_basename = os.path.basename(vid_path).split('.')[0]
    cap_basename = os.path.basename(cap_path).split('.')[0]
    assert vid_basename == cap_basename
    dst_path = os.path.join(outp_dir, vid_basename + out_ext)
    if force and os.path.isfile(dst_path):
      _getSharedLogger().warning("FORCE MODE ENABLED!!!! Overwriting existing file: '%s'...", dst_path)

    # Extract, prune, and filter captions.
    captions = _caption.extract_captions(cap_path)
    captions = _caption.prune_and_filter_captions(captions)

    # Extract face-dots.
    landmark_seqs = []
    video_reader = _video.VideoReader(vid_path)

    for (start, end), cap in captions.items():
      start_frame = video_reader.get_frame_idx(start)
      end_frame = video_reader.get_frame_idx(end)
      frames = video_reader.genFrames(start_frame, end_frame)
      landmark_seq = _face.get_landmarks(frames)
      landmark_seqs.append(landmark_seq)





def main(args):
  global _logger
  args = _cmd.parseArgsForClassOrScript(generate_dataview)
  varsArgs = vars(args)
  verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
  _getSharedLogger(verbosity=verbosity).info("Passed arguments: '{}'".format(varsArgs))
  generate_dataview(**varsArgs)

if __name__ == '__main__':
  main(sys.argv)