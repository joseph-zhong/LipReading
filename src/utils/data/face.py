#!/usr/bin/env python3
"""
face.py
---

Face detection and landmarking utilities.

Relies on `dlib` for face-detection and `PRNet + PyTorch` for landmarking.

"""

import os

from src.utils.utility import _util

def _getSharedLogger():
  return _util.getLogger(os.path.basename(__file__).split('.')[0])

def get_landmarks(vid_path, start, end):
  pass