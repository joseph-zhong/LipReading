#!/usr/bin/env python3
"""
time.py
---

Time utilities.
"""

def get_secs(time_str):
  """ Extracts the number of seconds a string represents while formatted as
  ```
  HH:MM:SS
  ```
  """
  h, m, s = time_str.split(':')
  return float(h) * 3600 + float(m) * 60 + float(s)