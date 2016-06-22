# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-05 21:20:57
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-06-22 16:04:10

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

__all__ = ['tensorflow_arch', 'theano_arch']

import os
import json
import sys
import numpy as np
import tunas.core.env as env

if env.ARCH == None:
  _tmp_base_ = os.path.expanduser('~') + '/.tunas/'
  if not os.path.exists(_tmp_base_):
    os.makedirs(_tmp_base_)

  _config_fpath_ = _tmp_base_ + 'config.json'
  if not os.path.exists(_config_fpath_):
  # write default configuration to config.json
    config = {'floatx' : 'float32', 'eps' : '1e-7', 'arch' : 'tensorflow'}
    env.EPS = np.asarray((config['eps']), env.FLOATX)
    env.FLOATX = 'float32'
    env.ARCH = 'tensorflow'

    print('First import tunas, init .tunas in your home directory,', 
      'using Tensorflow architecture.')
    with open(_config_fpath_, 'w') as f:
      f.write(json.dumps(config) + '\n')
      f.close()
  else:
    with open(_config_fpath_, 'r') as f:
      config = json.load(f)
      if config['floatx'] not in {'float32', 'float64'}:
        raise Exception('type ' + config['floatx'] + ' is not valid.')
      env.FLOATX = config['floatx']
      env.EPS = np.asarray((config['eps']), env.FLOATX)
      env.ARCH = config['arch']
      f.close()

if env.ARCH == 'tensorflow': 
  try:
    import tensorflow as tf
    if not tf.test.is_built_with_cuda():
      print('GPU not supported on this session.')
    if env.SESSION == None:
      env.SESSION = tf.Session()
  except:
    raise ImportError("Need Tensorflow as arch dependencies,",
      " and cudnn installed. Please check the configuration.")
  from .tensorflow_arch.ops import *
  from .tensorflow_arch.tensor import *
    
elif env.ARCH == 'theano':
  try:
    import theano
    import theano.tensor as T
    if not theano.sandbox.cuda.cuda_enabled:
      print('GPU not supported on this session.')
  except:
    raise ImportError("Need Theano as arch dependencies, and cudnn installed. Please check the configuration.")
  from .theano_arch.ops import *
  from .theano_arch.tensor import *
else:
  raise ImportError('Unknown architecture. Please check your configuration file, or simply delete .tunas directory.')
