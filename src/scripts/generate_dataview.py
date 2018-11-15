#!/usr/bin/env python3
"""
generate_dataview.py
---

This reads the available video and caption files available and generates the corresponding dataviews.

A dataview is a dense table of data input and label pairs.
For our purposes, we will generate a table for each video-caption pair as follows:

| idx  |  start, end  |  face_frames     |  face_lmk_seq         | face_vtx_seq         | caption text |
| ---- | ------------ | ---------------- | --------------------- | -------------------- | ------------ |
| `i`  | `(s_i, e_i)` | `(frames, h, w)` | `(frames, lmks, yxz)` | `(frames, vtx, xyz)` |  `"str...."` |

Note, the face landmarks are landmarks with coordinates relative to the face frame, which are take from the raw
landmarks which are coordinates relative to the full frame.

There are 68 canonical face landmarks, and 45128 total face vertices in the point cloud.

Each video will have a corresponding table of caption-windows -> landmark sets, where the data is sequenced
based on the provided caption set splits, which correspond to a unique frame start and end, which will be used
to extract the unique landmarks extracted during the caption.

"""
import os
import sys
import glob
import time
import collections
import traceback

import numpy as np

import src.utils.data.video as _video
import src.utils.data.caption as _caption
import src.utils.data.face as _face

import src.utils.cmd_line as _cmd
import src.utils.utility as _util

_white = (255, 255, 255)
_logger = None

def _getSharedLogger(verbosity=_util.DEFAULT_VERBOSITY):
  global _logger
  if _logger is None:
    _logger = _util.getLogger(os.path.basename(__file__).split('.')[0], verbosity=verbosity)
  return _logger

def generate_dataview(
    inp="StephenColbert/nano1",
    outp_dir="StephenColbert/nano1",
    vid_ext=".mp4",
    cap_ext=".vtt",
    out_ext=".npy",
    timedelay=0,
    force=False,
    visualmode=False,
):
  """ Generates dataviews for the given input file or directory.

  :param inp: Input video/caption filename (without extension), or directory of video of video/captions.
    Videos must have extensions of 'vid_ext' and captions 'cap_ext' with coinciding names.
  :param outp_dir: Output directory for the datasets.
  :param timedelay: Hyperparameter to add a delay to the input landmarks.
    See section 3.1.1 Recurrent Neural Network for a description in
    http://grail.cs.washington.edu/projects/AudioToObama/siggraph17_obama.pdf
    TODO: Proper implementation and analysis, TO BE DECIDED.
  :param force: Flag to overwrite existing files.
  :param visualmode: Flag to visualize the dataview against the video frames.
    REVIEW josephz: This implementation is currently totally ass and should be deprecated now.
  """
  # Standardize directories.
  inp = _util.getRelRawPath(inp)
  outp_dir = _util.getRelDatasetsPath(outp_dir)

  # Glob corresponding video and caption files.
  vid_glob = os.path.join(inp, '*' + vid_ext)
  cap_glob = os.path.join(inp, '*' + cap_ext)
  vid_paths = sorted(glob.glob(vid_glob))
  cap_paths = sorted(glob.glob(cap_glob))
  assert len(vid_paths) == len(cap_paths) > 0

  ts = time.time()
  _getSharedLogger().info(
      "Processing '%d' number of videos", len(vid_paths))
  for i, vid_cap_path in enumerate(zip(vid_paths, cap_paths)):
    # Get inp_paths.
    vid_path, cap_path = vid_cap_path
    _getSharedLogger().info(
      "\tVideo (%4d/%4d): Extracting captions and landmarks for '%s'",
      i, len(vid_paths)-1, vid_path)

    # Get dst_path.
    vid_basename = os.path.basename(vid_path).split('.')[0]
    cap_basename = os.path.basename(cap_path).split('.')[0]
    assert vid_basename == cap_basename
    dst_path = os.path.join(outp_dir, vid_basename)
    _getSharedLogger().info(
      "\tVideo (%4d/%4d): Writing dataview to '%s'",
      i, len(vid_paths)-1, dst_path)

    # If force mode, or file doesn't exist, process video, else continue.
    if force or not os.path.isfile(dst_path):
      if force and os.path.isfile(dst_path):
        _getSharedLogger().warning(
          "\tVideo (%4d/%4d): FORCE MODE ENABLED!!!! Overwriting existing file: '%s'...",
          i, len(vid_paths) - 1, dst_path)
      _getSharedLogger().info(
        "\tVideo (%4d/%4d): Writing dataview to '%s'",
        i, len(vid_paths) - 1, dst_path)

      # Extract, prune, and filter captions.
      captions = _caption.extract_captions(cap_path)
      captions = _caption.prune_and_filter_captions(captions)

      # Extract face-dots for each caption as a dataview.
      dataview = _generate_dataview(vid_path, captions, timedelay=timedelay, visualmode=visualmode)

      # Save dataview.
      for col, rows in dataview.items():
        dst_col_path = os.path.join(dst_path, col + out_ext)
        if force or not os.path.isfile(dst_col_path):
          _getSharedLogger().info(
            "\tVideo (%4d/%4d): Writing '%s' dataview to '%s'",
              i, len(vid_paths) - 1, col, dst_col_path)
          _util.mkdirP(os.path.dirname(dst_col_path))
          np.save(dst_col_path, rows)
    else:
      _getSharedLogger().warning(
        "\tVideo (%4d/%4d): Skipping existing file: '%s'...", i, len(vid_paths)-1, dst_path)

  te = time.time()
  _getSharedLogger().info("Done writing dataviews! Took %0.3f seconds", te - ts)

# REVIEW josephz: How do I implement timedelay for this scenario? In the original implementation, the input
# audio signal was "delayed" by some ms, or the first `k` ms of input were cut along with the last corresponding
# `k` ms output. It is possible here that the opposite is true, where looking ahead in the mouth-shape will
# inform the model of the context? Or is even relevant? Or do we actually want to add frames on each side?
def _generate_dataview(vid_path, captions, timedelay=0, visualmode=False):
  """ Extracts landmarks that coincide with the captions. Returns a dataview for a particular video-caption pair
  for frames that correspond with valid captions that also coincide a reasonably detectable face.
  """
  assert os.path.isfile(vid_path)
  assert isinstance(captions, collections.OrderedDict) and len(captions) > 0

  dataview = collections.OrderedDict((col, [])
    for col in ('s_e', 'caption_faces', 'caption_raw_lmks', 'caption_face_lmks', 'caption_face_vtx', 'cap'))
  video_reader = _video.VideoReader(vid_path)

  # Iterate through each caption and extract the corresponding frames in (start, end).
  for cap_idx, s_e in enumerate(captions.keys()):
    start, end = s_e
    cap = captions[s_e]
    # caption_raw_lmks = []
    caption_face_lmks = []
    caption_face_vtx = []
    caption_faces = []

    # REVIEW josephz: How to apply timedelay here?
    start_frame = video_reader.get_frame_idx(start)
    end_frame = video_reader.get_frame_idx(end)
    frames = video_reader.genFrames(start_frame, end_frame)
    _getSharedLogger().info("\tCaption (%4d/%4d): Computing landmarks for '%d' frames",
      cap_idx + 1, len(captions) - 1, end_frame - start_frame)

    # For visualization, we need to have a rolling cache of the caption's frames.
    # REVIEW josephz: This really should be implemented in VideoReader.
    dequeue = None
    if visualmode:
      dequeue = collections.deque(maxlen=end_frame-start_frame+1)

    # For each corresponding frame, detect face, extract the 3D landmarks, and accumulate caption-lmks pairs.
    for i, frame in enumerate(frames):
      ts = time.time()
      assert len(frame.shape) == 3

      # Detect face and extract 3d landmarks.
      try:
        frame_face_rect = _face.detectMaxFaceRect(frame, times_to_upsample=1)
        frame_face, frame_face_rect_pad = _face.extractFace(frame, frame_face_rect, padding=0.3)
        frame_lmks, _ = _face.detect3dLandmarks(frame, rect=frame_face_rect)
        frame_vtx = _face.get3dVertices()
        frame_face_lmks = _face.getFace(frame_lmks, frame_face_rect_pad)
        frame_face_vtx = _face.getFace(frame_vtx, frame_face_rect_pad)

        # REVIEW josephz: deprecate this...
        if visualmode:
          import cv2
          import matplotlib.pyplot as plt
          for x, y, z in frame_face_vtx:
            cv2.circle(frame_face, (int(round(x)), int(round(y))), 1, z, thickness=-1)
          plt.imshow(frame_face)
          plt.show()

        # If error occurs in detecting face or landmarks, skip.
        # REVIEW josephz: How else to check that most of the landmarks are valid? Could also check the size of the
        # face to be greater than a certain box.
        if frame_lmks is None:
          _getSharedLogger().warning("\tFrame (%4d/%4d): No face or landmarks detected for caption frame=%d",
            start_frame + i, video_reader.getNumFrames() - 1, start_frame + i)
        else:
          # Otherwise accumulate landmarks and faces for the caption.
          # caption_raw_lmks.append(frame_lmks)
          caption_faces.append(frame_face)
          caption_face_lmks.append(frame_face_lmks)
          caption_face_vtx.append(frame_face_vtx)

          _getSharedLogger().info("\tFrame (%4d/%4d): Generated data example for caption frame=%d, took '%0.3f' seconds",
            start_frame + i, video_reader.getNumFrames() - 1, start_frame + i, time.time() - ts)

        # Accumulate frame cache for visualization.
        if visualmode:
          assert dequeue is not None
          dequeue.append(frame_face)
      except KeyboardInterrupt:
        _getSharedLogger().warning("\tFrame (%4d/%4d): KeyboardInterrupt, aborting...",
          start_frame + i, video_reader.getNumFrames() - 1)
        exit()
      except Exception as e:
        _getSharedLogger().error("\tFrame (%4d/%4d): Unxepected exception '%s', skipping caption...",
          start_frame + i, video_reader.getNumFrames() - 1, e)
        traceback.print_exc()
        break

    # Accumulate caption_lmks pair into dataview.
    dataview['s_e'].append(s_e)
    dataview['caption_faces'].append(caption_faces)
    # dataview['caption_raw_lmks'].append(caption_raw_lmks)
    dataview['caption_face_lmks'].append(caption_face_lmks)
    dataview['caption_face_vtx'].append(caption_face_vtx)
    dataview['cap'].append(cap)

    # Visualize caption-lmks pair.
    # REVIEW josephz: This is really absolutely shit and should be killed.
    if visualmode:
      import cv2
      import matplotlib.pyplot as plt
      import matplotlib.animation as ani

      fig = plt.figure()
      vis = list(dequeue)
      assert len(vis) == len(caption_face_lmks)
      ims = []
      for i, lmks in enumerate(caption_face_lmks):
        for x, y, z in lmks:
          cv2.circle(vis[i], (int(round(x)), int(round(y))), 1, _white, thickness=-1)
        ims.append((plt.imshow(vis[i]),))
      ani = ani.ArtistAnimation(fig, ims, interval=33, blit=True, repeat_delay=3000)
      print("Caption:", cap)
      plt.show()

  # Convert Python lists to np.ndarray, and return.
  for k, v in dataview.items():
    assert isinstance(v, list)
    dataview[k] = np.array(v)
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