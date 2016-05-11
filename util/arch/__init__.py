# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-05 21:20:57
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-11 20:01:56

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import json
import sys

global _EPS_
global _FLOATX_
global _ARCH_
global _SESSION_

__all__ = ['expr', 'func', 'math', 'optimization', 'commmon']

sys.path.append('.')
# read configure file
_tmp_base_ = '~/.tunas/'
if not os.path.exists(_tmp_base_):
    os.makedirs(_tmp_base_)

_config_fpath_ = _tmp_base_ + 'config.json'
if not os.path.exists(_config_fpath_):
    # write default configuration to config.json
    import tensorflow as tf
    config = {'floatx' : 'float32', 'eps' : '1e-7', 'arch' : 'tensorflow'}
    _EPS_ = 1e-7
    _FLOATX_ = 'float32'
    _ARCH_ = 'tensorflow'
    _SESSION_ = tf.Session()
    with open(_config_fpath_, 'w') as f:
        f.write(json.dumps(config) + '\n')
else:
    with open(_config_fpath_, 'r') as f:
        config = json.load(f)
        if config['floatx'] not in {'float32', 'float64'}:
            raise Exception('type ' + config['floatx'] + ' is not valid.')
        _FLOATX_ = config['floatx']
        _EPS_ = config['eps']
        _ARCH_ = config['arch']
        _SESSION_ = None

if _ARCH_ == 'tensorflow': 
    try:
        import tensorflow as tf
    except:
        raise ImportError("Need Tensorflow as arch dependencies, and cudnn installed. Please check the configuration.")
    from .tensorflow.expr import *
    from .tensorflow.func import *
    from .tensorflow.math import *
    from .tensorflow.optimization import *
    from .common import *
        
elif _ARCH_ == 'theano':
    try:
        import theano
        import theano.tensor as T
        if not theano.sandbox.cuda.cuda_enabled:
            raise ImportError(
                    "requires GPU support -- see http://lasagne.readthedocs.org/en/"
                    "latest/user/installation.html#gpu-support")
        from util.arch.theano import *
    except:
        raise ImportError("Need Theano as arch dependencies, and cudnn installed. Please check the configuration.")
else:
    raise ImportError('Unknown architecture. Please check your configuration file, or simply delete .tunas directory.')
