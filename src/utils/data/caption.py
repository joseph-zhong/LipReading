#!/usr/bin/env python3
"""
caption.py
---

Caption utilities.

"""

import os
import re
import collections

import pycaption

from src.utils.time import _time
from src.utils.utility import _util

_patterns = (
    r"\(.*\)",
)

_conditions = (
  lambda x: len(x.split()) <= 2,
  lambda x: len(x) <= 8,
)

def _getSharedLogger():
  return _util.getLogger(os.path.basename(__file__).split('.')[0])

def extract_captions(cap_fname, lang='en-US'):
  """ Reads a list of captions and returns an ordered dictionary of {(start_time, end_time) -> "caption"}
  with time in units of seconds. The time is extracted by assuming each caption is prefixed by

  ```
  "00:00:00.467 --> 00:00:02.836..."
  ```

  :param cap_fname: VTT subtitle file to read from.
  """
  assert os.path.isfile(cap_fname)
  _getSharedLogger().info("Reading captions from '%s'", cap_fname)
  reader = pycaption.WebVTTReader()
  res = collections.OrderedDict()
  with open(cap_fname) as fin:
    captions_raw = fin.read()
    assert reader.detect(captions_raw), "Malformed file: '{}'".format(cap_fname)

    caption_set = reader.read(captions_raw)
    assert not caption_set.is_empty(), "Empty VTT file: '{}'".format(cap_fname)
    # REVIEW josephz: We'll need to check what other possibilities there are.
    assert lang in caption_set.get_languages()

    captions = caption_set.get_captions(lang=lang)
    assert len(captions) > 0

  _getSharedLogger().info("Detected '%s'", len(captions))
  for cap in captions:
    start_end, cap_raw = cap[:29], cap[29:]
    start, end = start_end.split(' --> ')
    start = _time.get_secs(start)
    end = _time.get_secs(end)
    res[(start, end)] = cap_raw.trim()
  assert len(res) == len(captions)
  return res

def prune_and_filter_captions(captions, patterns=None, conditions=None, union=True):
  """ Cleans a dictionary of time-sequenced captions based on regex patterns to prune invalid tokens within
  captions, as well as filtering conditions to delete captions entirely.

  i.e. The following regex patterns will match on any characters encapsulated in opening and closing parentheses.
  ```
  patterns = [
    r"\(.*\)",
  ]
  ```

  Furthermore, when any of the following conditions match the caption, it will be deleted.
  ```
  filter = [
    lambda x: len(x.split()) <= 2,
    lambda x: len(x) <= 8,
  ]
  ```

  :param captions: Dictionary of captions to prune and filter.
  :param patterns: Regex patterns to prune caption tokens.
  :param conditions: Boolean conditions to filter captions.
  """

  if conditions is None:
    conditions = _conditions
  if patterns is None:
    patterns = _patterns
  regex = re.compile("|".join(patterns))

  # Remove matched patterns within captions.
  for k, cap_raw in captions.items():
    captions[k] = regex.sub('', cap_raw)

  # Filter captions based on caption condition filters.
  fn = any if union else all
  res = collections.OrderedDict(
    (k, cap) for k, cap in captions
      if fn(cond(cap) for cond in conditions))
  return res
