# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-05 21:20:57
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-12-05 16:54:17

from __future__ import absolute_import, print_function, division

import os
import json
import sys
import numpy as np
from .env import set_epsilon, get_epsilon
from .env import set_floatX, get_floatX
from .env import set_arch, get_arch
from .env import set_session, get_session
from .env import set_device, get_device

fconfig = os.path.expanduser('~') + '/.tunas'

if not os.path.exists(fconfig):
  raise Exception("Please run init.py")
else:
  with open(fconfig, 'r') as f:
    config = json.load(f)
    if config['floatx'] not in {'float32', 'float64'}:
      raise Exception('type ' + config['floatx'] + ' is not valid.')

set_floatX(config['floatx'])
set_epsilon(np.asarray((config['eps']), config['floatx']))
set_arch(config['arch'])
set_device(config['device'])

if get_arch() == 'tensorflow': 
  try:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['device'])
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    if not tf.test.is_built_with_cuda():
      print('GPU not supported on this session.')
    if get_session() == None:
      set_session(tf.Session(config = config))
  except:
    raise ImportError("Need Tensorflow as arch dependencies,",
      " and cudnn installed. Please check the configuration.")
  from .tensorflow_arch import *
    
elif get_arch() == 'theano':
  gpus = ''.join(['gpu' + k + ',' for k in config['device'].split(',')])
  os.environ['THEANO_FLAGS'] = 'device=' + gpus + 'floatX=' + config['floatx']
  try:
    import theano
    import theano.tensor as T
    if not theano.sandbox.cuda.cuda_enabled:
      print('GPU not supported on this session.')
  except:
    raise ImportError("Need Theano as arch dependencies, and cudnn installed. Please check the configuration.")
  from .theano_arch import *
else:
  raise ImportError('Unknown architecture. Please check your configuration file, or simply delete .tunas directory.')
