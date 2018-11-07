#!/usr/bin/env python3
"""
face.py
---

Face detection and landmarking utilities.

Relies on `dlib` for face-detection and `PRNet + PyTorch` for landmarking.

"""

import os
import imageio
import collections

import numpy as np

import src.utils.utility as _util

from extern.PRNet.api import PRN

_prn = None

def _getSharedLogger():
  return _util.getLogger(os.path.basename(__file__).split('.')[0])

def _getSharedPrn(is_dlib=True, gpuIds="0"):
  global _prn
  if _prn is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpuIds
    _prn = PRN(is_dlib=is_dlib)
  return _prn

def get_landmarks(frames, is_dlib=True, gpuIds="0"):
  """ Extracts landmarks from the largest detected face in each frame of the provided video.

  :param is_dlib:
  :param gpuIds:
  """
