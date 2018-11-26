#!/usr/bin/env python3
"""
caption.py
---

Caption utilities.

"""

import os
import re
import collections
import unicodedata

import pycaption

import src.utils.time as _time
import src.utils.utility as _util

_patterns = (
    r"\(.*\)",
    r"<[^>]*>",
    r"\[.*\]",
    r"\{.*\}",
    r"stephen:",
    r">>",
)

_conditions = (
  lambda x: len(x.split()) <= 2,
  # lambda x: len(x) <= 8,
)

def _getSharedLogger():
  return _util.getLogger(os.path.basename(__file__).split('.')[0])

def extract_captions(cap_fname, lang='en-US'):
  """ Reads a list of captions and returns an ordered dictionary of {(start_time, end_time) -> "caption"}
  with time in units of seconds.

  :param cap_fname: VTT subtitle file to read from. Produces Caption sets with text, and times in microseconds.
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

  _getSharedLogger().info("Detected '%s' captions...", len(captions))
  for c in captions:
    cap_raw = c.get_text()
    start = _time.micros_to_sec(c.start)
    end = _time.micros_to_sec(c.end)
    res[(start, end)] = cap_raw.strip()
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
  :param patterns: Regex patterns to prune caption tokens. Should be lowercase only.
  :param conditions: Boolean conditions to filter captions.
  """
  if conditions is None:
    conditions = _conditions
  if patterns is None:
    patterns = _patterns
  regex = re.compile("|".join(patterns))

  # Remove matched patterns within captions.
  for k, cap_raw in captions.items():
    cap_raw = regex.sub('', cap_raw).strip()
    cap_raw = cap_raw.replace('\n', ' ')
    cap_raw = cap_raw.lower()
    cap_raw = unicodedata.normalize(u'NFKD', cap_raw).encode('ascii', 'ignore').decode('utf8')
    captions[k] = cap_raw

  # Filter captions based on caption condition filters.
  fn = any if union else all
  res = collections.OrderedDict(
    (k, cap) for k, cap in captions.items()
      if not fn(cond(cap) for cond in conditions))
  return res
