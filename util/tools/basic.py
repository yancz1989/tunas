# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-05 21:20:45
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-06 14:06:08

import os
import sys
import time
import h5py
import shutil

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
    # @path, string
    if os.isfile(path):
        os.remove(path)
    else:
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

