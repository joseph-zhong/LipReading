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

def ms_to_sec(ms):
  return float(ms) / 1000

def micros_to_sec(mics):
  return float(mics) / 1000 / 1000

def sec_to_ms(sec):
  return sec * 1000 * 1000