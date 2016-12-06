# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-05 21:20:45
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-12-05 16:40:32

import os
import os.path
import sys
import time
import h5py
import shutil
import logging
from enum import Enum

class LogMode(Enum):
  warning = 1
  debug = 2
  info = 3
  error = 4
  critical = 5

class Logger:
  def __init__(self):
    logging.basicConfig(stream = sys.stdout, level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

  def log(self, mode, str):
    if mode == LogMode.warning:
      logging.warning(str)
    elif mode == LogMode.debug:
      logging.debug(str)
    elif mode == LogMode.info:
      logging.info(str)
    elif mode == LogMode.error:
      logging.error(str)
    elif mode == LogMode.critical:
      logging.critical(str)


# os support
def cd(path):
  # @path: path string.
  os.chdir(dir)

def ls(path, hide_dot = True):
  # @path, string.
  # @hide_dot, bool, whether or not show files and directories start with dot.
  items = os.listdir(path)
  files = []
  dirs = []
  for item in items:
    if os.path.isfile(item):
      files.append(item)
    else:
      dirs.append(item)
  if hide_dot:
    files = [file for file in files if file[0] != '.']
    dirs = [dir for dir in dirs if dir[0] != '.']
  return {'files' : files, 'dirs' : dirs}

def mv(src, dst):
  # @src, string, source file or directory.
  # @dst, string, destination file or directory. If dst already exists, rename will be used.
  shutil.move(src, dst)

def rm(path):
  if os.path.isfile(path):
    os.path.remove(path)
  elif os.path.isdir(path):
    shutil.rmtree(path)

def cp(src, dst, symlink = False, ignores = []):
  '''
    to copy file or directories.
    @src, string, source file or directory.
    @dst, string, destination file or directory
    @symlink, bool, whether ignore symlinks
    @ignores, list, ignore patterns, used as parameter for ignore_patterns.
  '''
  if os.isfile(src):
    shutil.copy(src, dst)
  else:
    shutil.copytree(src, dst, symlink, shutil.ignore_patterns(ignores))

def countf(path):
  # count number of files in path
  return len(ls(path)['files'])

def mkdir(dname):
  if not os.path.exists(dname):
    os.makedirs(dname)

def inst2list(x):
  if x == None:
    l = None
  elif isinstance(x, list):
    l = x
  else:
    l = [x]
  return l

def dict2list(dict):
  return [dict[key] for key in dict]

