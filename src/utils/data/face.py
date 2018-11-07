#!/usr/bin/env python3
"""
face.py
---

Face detection and landmarking utilities.

Relies on `dlib` for face-detection and `PRNet + PyTorch` for landmarking.

"""

import os
import dlib
import imageio
import collections

import numpy as np

import src.utils.utility as _util

from src.models.face.prnet import PRN

_mouth = slice(48, 68)
_right_eyebrow = slice(17, 22)
_left_eyebrow = slice(22, 27)
_right_eye = slice(36, 42)
_left_eye = slice(42, 48)
_nose = slice(27, 35)
_jaw = slice(0, 17)

_face_pts = 68
_mouth_pts = 20

# Shared face detector, landmarker, and reference 3D Model for frontalization.
_face_landmarks_path = "./data/weights/dlib/shape_predictor_68_face_landmarks.dat"

_prn = None
_detector = None
_landmarker = None

def _getSharedLogger():
  return _util.getLogger(os.path.basename(__file__).split('.')[0])

def _getSharedPrn(is_dlib=True, gpuIds="0"):
  global _prn
  if _prn is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpuIds
    _prn = PRN(is_dlib=is_dlib)
  return _prn

def _getSharedDetector():
  global _detector
  if _detector is None:
    _detector = dlib.get_frontal_face_detector()
  return _detector

def _getSharedLandmarker():
  global _landmarker
  if _landmarker is None:
    _landmarker = dlib.shape_predictor(_face_landmarks_path)
  return _landmarker

def detectFaces(img, times_to_upsample=2):
  rects = _getSharedDetector()(img, times_to_upsample)
  return rects

def detectMaxFace(img, times_to_upsample=2):
  rects = detectFaces(img, times_to_upsample=times_to_upsample)
  maxRectIdx = np.argmax(x.height() * x.width() for x in rects)
  return rects[maxRectIdx]

def computeMaxLandmarks(img, times_to_upsample=2):
  """ Computes max face landmarks. """
  maxRect = detectMaxFace(img, times_to_upsample=times_to_upsample)
  lmks = _getSharedLandmarker()(img, maxRect)
  assert lmks is not None, "Failed to compute landmarks for image of shape: '{}'".format(img.shape)
  return lmks

def get_3d_landmarks(img, is_dlib=True, gpuIds="0"):
  """ Extracts 3D landmarks from the largest detected face in each frame of the provided video.
  """
  assert len(img.shape) == 3
  # import matplotlib.pyplot as plt
  # plt.imshow(img)
  # plt.show()

  pos_map = _getSharedPrn().process(img)
  if pos_map is None:
    return None
  lmks3d = _getSharedPrn(is_dlib=is_dlib, gpuIds=gpuIds).get_landmarks(pos_map)
  if lmks3d is None:
    _getSharedLogger().warning("No face detected!")
  return lmks3d
