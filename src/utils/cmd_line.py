#!/usr/bin/env python
"""
cmd_line.py
---

CMD Line parsing utilities.

"""
import argparse
import collections
import inspect
import logging
import subprocess
from types import GeneratorType

import src.utils.utility as _util

_logger = _util.getLogger("CMD Line")

def runCmd(cmd, logger=None, stopOnFail=True):
  if logger is None:
    logger = _logger
  else:
    assert isinstance(logger, logging.Logger)

  logger.info("Running '%s'", cmd)
  # output = subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT, shell=True)
  # print output
  # ret = subprocess.call(cmd.split(), shell=True)
  # if ret != 0:
  #   logger.error("'%s' returned with error code: '%s'", cmd, ret)
  #   logger.debug("Traceback: '%s'", traceback.format_exc())
  #   if stopOnFail:
  #     sys.exit(ret)
  # else:
  #   logger.info("'{}' Success!".format(cmd))
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
  proc_stdout = process.communicate()[0].strip()
  print(proc_stdout)
  logger.info("Completed running '%s", cmd)

def _str_to_bool(s):
  """Convert string to bool (in argparse context)."""
  if s.lower() not in ['true', 'false']:
    raise ValueError('Need bool; got %r' % s)
  return {'true': True, 'false': False}[s.lower()]

def add_boolean_argument(parser, name, default=False):
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
    '--' + name,
    nargs='?',
    default=default,
    const=True,
    type=_str_to_bool)
  group.add_argument('--no' + name,
    dest=name,
    action='store_false')

# REVIEW josephz: Deprecate this. Merge this into `parseArgsForClassOrScript`.
def parseArgsForCustom(defaultArgs):
  assert isinstance(defaultArgs, collections.OrderedDict)
  parser = argparse.ArgumentParser()
  for k, v in defaultArgs.iteritems():
    if isinstance(v, bool):
      parser.add_argument("--" + k, default=v, action='store_true')
    else:
      parser.add_argument("--" + k, default=v, type=type(v) if v is not None else str)

  parser.add_argument("-v", "--verbosity",
    default=_util.DEFAULT_VERBOSITY,
    type=int,
    help="Verbosity mode. Default is 4. "
         "Set as "
         "0 for CRITICAL level logs only. "
         "1 for ERROR and above level logs "
         "2 for WARNING and above level logs "
         "3 for INFO and above level logs "
         "4 for DEBUG and above level logs")
  argv = parser.parse_args()
  argToDefaults = vars(argv)

  if argv.verbosity > 0 or argv.help:
    print()
    print("Arguments and corresponding default or set values")
    for argName, argDefault in argToDefaults.iteritems():
      print("\t{}={}".format(argName, argDefault if argDefault is not None else ""))

  return argv

def parseArgsForClassOrScript(fn):
  assert inspect.isfunction(fn) or inspect.ismethod(fn)

  spec = inspect.getargspec(fn)

  parser = argparse.ArgumentParser()
  for i, arg in enumerate(spec.args):
    if arg == 'self' or arg == 'logger':
      continue

    # If index is greater than the last var with a default, it's required.
    numReq = len(spec.args) - len(spec.defaults)
    required = i < numReq
    default = spec.defaults[i - numReq] if not required else None
    # By default, args are parsed as strings if not otherwise specified.
    if isinstance(default, bool):
      parser.add_argument("--" + arg, default=default, action='store_true')
    elif isinstance(default, (tuple, list, GeneratorType)):
      parser.add_argument("--" + arg, default=default, nargs="+", help="Tuple of " + arg, required=False)
    else:
      parser.add_argument("--" + arg, default=default, type=type(default) if default is not None else str)

  parser.add_argument("-v", "--verbosity",
    default=_util.DEFAULT_VERBOSITY,
    type=int,
    help="Verbosity mode. Default is 4. "
         "Set as "
         "0 for CRITICAL level logs only. "
         "1 for ERROR and above level logs "
         "2 for WARNING and above level logs "
         "3 for INFO and above level logs "
         "4 for DEBUG and above level logs")
  argv = parser.parse_args()
  argsToVals = vars(argv)

  if argv.verbosity > 0 or argv.help:
    docstr = inspect.getdoc(fn)
    assert docstr is not None, "Please write documentation :)"
    print()
    print(docstr.strip())
    print()
    print("Arguments and corresponding default or set values")
    for arg in spec.args:
      if arg == 'self' or arg == 'logger' or arg not in argsToVals:
        continue
      print("\t{}={}".format(arg, argsToVals[arg] if argsToVals[arg] is not None else ""))
    print()

      # parser.print_help()

  return argv