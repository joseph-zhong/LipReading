#!/usr/bin/env python3
"""
face.py
---

Face detection and landmarking utilities.

Relies on `dlib` for face-detection and `PRNet + PyTorch` for landmarking.

"""

import os

import time
import dlib
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

def detectFaceRects(img, times_to_upsample=2):
  rects = _getSharedDetector()(img, times_to_upsample)
  return rects

def detectMaxFaceRect(img, times_to_upsample=2):
  _getSharedLogger().debug("Detecting maximum face rect for img of shape='%s'...", img.shape)
  ts = time.time()
  rects = detectFaceRects(img, times_to_upsample=times_to_upsample)
  _getSharedLogger().debug("Done! Took '%0.3f' seconds...", time.time() - ts)
  assert len(rects) > 0
  maxRectIdx = np.argmax(x.height() * x.width() for x in rects)
  rect = rects[maxRectIdx]
  rect = rect.left(), rect.right(), rect.top(), rect.bottom()
  return rect

def _applyPadding(dims, rect, padding):
  """ Apply padding to each side of a rectangle based on width and height percentage. """
  assert len(dims) == 3
  img_h, img_w, _ = dims

  left, right, top, bottom = rect
  box_h = bottom - top
  box_w = right - left

  # Apply bounded padding.
  left = max(0, left - int(padding * box_w))
  right = min(img_w, right + int(padding * box_w))
  top = max(0, top - int(padding * box_h))
  bottom = min(img_h, bottom + int(padding * box_h))
  return left, right, top, bottom

def detectMaxFace(img, rect=None, times_to_upsample=2, padding=0.2):
  assert len(img.shape) == 3
  if rect is None:
    rect = detectMaxFaceRect(img, times_to_upsample=times_to_upsample)

  if padding is not None:
    left, right, top, bottom = _applyPadding(img.shape, rect, padding)
  else:
    left, right, top, bottom = rect
  return img[top:bottom, left:right, :]

def extractFace(img, rect, padding=None):
  assert len(img.shape) == 3
  assert isinstance(rect, tuple) and len(rect) == 4

  # Apply padding.
  if padding is not None:
    assert 0 < padding <= 0.5
    rect = _applyPadding(img.shape, rect, padding)

  left, right, top, bottom = rect
  res = img[top:bottom, left:right, :]
  assert all(x > 0 for x in res.shape)
  return res, rect

def computeMaxLandmarks(img, times_to_upsample=2):
  """ Computes max face landmarks. """
  maxRect = detectMaxFaceRect(img, times_to_upsample=times_to_upsample)
  lmks = _getSharedLandmarker()(img, maxRect)
  assert lmks is not None, "Failed to compute landmarks for image of shape: '{}'".format(img.shape)
  return lmks

def detect3dLandmarks(img, rect=None, is_dlib=False, gpuIds="0"):
  """ Extracts 3D landmarks from the largest detected face in each frame of the provided video.
  """
  assert len(img.shape) == 3
  _getSharedLogger().info("Computing Position Map for img of shape='%s'...", img.shape)
  ts = time.time()
  pos_map, inp_img = _getSharedPrn(is_dlib=is_dlib, gpuIds=gpuIds).process(img, image_info=rect)
  _getSharedLogger().debug("Done! Took '%0.3f' seconds", time.time() - ts)
  if pos_map is None:
    return None
  lmks3d = _getSharedPrn(is_dlib=is_dlib, gpuIds=gpuIds).get_landmarks(pos_map)
  if lmks3d is None:
    _getSharedLogger().warning("No face detected!")
  return lmks3d, inp_img

def detect3dVertices(img, rect=None, is_dlib=True, gpuIds="0"):
  assert len(img.shape) == 3
  _getSharedLogger().info("Computing Position Map for img of shape='%s'...", img.shape)
  ts = time.time()
  pos_map, inp_img = _getSharedPrn(is_dlib=is_dlib, gpuIds=gpuIds).process(img, image_info=rect)
  _getSharedLogger().debug("Done! Took '%0.3f' seconds", time.time() - ts)
  if pos_map is None:
    return None
  lmks3d = _getSharedPrn(is_dlib=is_dlib, gpuIds=gpuIds).get_landmarks(pos_map)
  if lmks3d is None:
    _getSharedLogger().warning("No face detected!")
  return lmks3d, inp_img

def get3dLandmarks():
  assert _getSharedPrn().pos is not None
  pos_map = _getSharedPrn().pos
  lmks3d = _getSharedPrn().get_landmarks(pos_map)
  return lmks3d

def get3dVertices():
  assert _getSharedPrn().pos is not None
  pos_map = _getSharedPrn().pos
  vertices3d = _getSharedPrn().get_vertices(pos_map)
  return vertices3d

def getFace(inp, rect):
  """ Converts raw landmarks to face landmarks, relative to a face rectangle.
  This is simply a translation of all the xy coordinates: leftwards by the rectangle's left coordinate, and upwards
  by the rectangles top coordinate.
  """
  assert len(inp.shape) == 2 and inp.shape[1] == 3
  assert isinstance(rect, tuple) and len(rect) == 4
  left, right, top, bottom = rect
  res = inp.copy()
  res[:, 0] -= left
  res[:, 1] -= top
  return res
