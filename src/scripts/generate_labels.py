#!/usr/bin/env python3
"""
generate_labels.py
---

Generates a JSON list of all unique characters for a given dataset.

"""

import glob
import json
import os
import sys
import tqdm as _tqdm

import src.utils.utility as _util
import src.utils.cmd_line as _cmd
from utils.data.caption import extract_captions, prune_and_filter_captions

_logger = _util.getLogger("CMD Line")

def _getSharedLogger(verbosity=_util.DEFAULT_VERBOSITY):
    global _logger
    if _logger is None:
        _logger = _util.getLogger(os.path.basename(__file__).split('.')[0], verbosity=verbosity)
    return _logger


def _gen_from_dataset(dataset_path, cap_ext):
    assert os.path.isdir(dataset_path)
    raise NotImplementedError


def _gen_from_raw(raw_path, cap_ext):
    assert os.path.isdir(raw_path)

    chars = set()
    for video in _tqdm.tqdm(glob.glob(os.path.join(raw_path, '*' + cap_ext))):
        captions = extract_captions(video)
        captions = prune_and_filter_captions(captions)

        for caption in captions.values():
            chars.update(caption)
    return sorted(chars)


def generate_labels(data, output="labels.json", cap_ext=".vtt", use_raw=True):
    """
    Generates an ordered list of labels, unique characters used in video captions, from
    the specified data.

    :param data: Either dataset or raw data name.
    :param output: File in data directory to which the labels should be written, or - for stdout.
    :param cap_ext: File extension for subtitle files.
    :param use_raw: True to use raw, else uses dataset.
    """
    data_path = _util.getRelRawPath(data) if use_raw else _util.getRelDatasetsPath(generate_labels)

    if use_raw:
        labels = _gen_from_raw(data_path, cap_ext)
    else:
        labels = _gen_from_dataset(data_path, cap_ext)

    if output == "-":
        json.dump(labels, sys.stdout)
    else:
        with open(os.path.join(data_path, output), "w") as out:
            json.dump(labels, out)


def main():
    global _logger
    args = _cmd.parseArgsForClassOrScript(generate_labels)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
    _getSharedLogger(verbosity=verbosity).info("Passed arguments: '{}'".format(varsArgs))
    generate_labels(**varsArgs)


if __name__ == '__main__':
    main()
