#!/usr/bin/env python

"""
Uility functions.
"""

import os
import shutil
import logging

DEFAULT_VERBOSITY = 4

_cachedImDir = None
_cachedIm2Dir = None

_logger = None
_LOGGING_FORMAT = "[%(asctime)s %(levelname)5s %(filename)s %(funcName)s:%(lineno)s] %(message)s"
# REVIEW josephz: How do I enable file-logging as well?
logging.basicConfig(format=_LOGGING_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

def getLogger(name, level=logging.DEBUG, verbosity=DEFAULT_VERBOSITY):
  level = max(level, logging.CRITICAL - 10 * verbosity)

  logger = logging.getLogger(name)
  logger.setLevel(level)
  return logger

def _getPathFromEnv(envVar):
  path = os.getenv(envVar, None)
  assert path is not None, \
    "Environment variable '{}' not found: " \
    "please check project installation and ~/.bashrc".format(envVar)
  return path

def _getUtilityLogger():
  global _logger
  if _logger is None:
    _logger = getLogger("Utility")
  return _logger

def mkdirP(path):
  if not os.path.exists(path):
    os.makedirs(path)

def touch(path):
  with open(path, 'a'):
    os.utime(path, None)

def mv(src, dst, mkdirMode=True, force=False):
  """ Moves src to dst as if `mv` was used. Both src and dst are relative to root,
  which is set to the 'data path' by default. With the mkdir option, we enforce
  the dst to be a path and we support "move to dir" behavior. Otherwise we support
  "move to dir and rename file" behavior.
  """
  assert os.path.exists(src), "'{}' not found".format(src)

  # In mkdir mode, we enforce the dst to be a path to allow "move to dir" behavior.
  # Otherwise we are supporting "move to dir and rename" behavior.
  if not dst.endswith('/') and mkdirMode:
    dst += '/'
  dstHeadPath, _ = os.path.split(dst)
  mkdirP(dstHeadPath)

  if os.path.isdir(dst):
    _getUtilityLogger().info("Moving '{}' into directory '{}'".format(src, dst))
  else:
    _getUtilityLogger().info("Renaming '{}' to '{}'".format(src, dst))

  if force:
    shutil.copy(src, dst)
  else:
    shutil.move(src, dst)