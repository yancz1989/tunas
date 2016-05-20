# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-05 21:20:57
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-20 20:35:12

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

__all__ = ['expr', 'func', 'math', 'optimization']

import os
import json
import sys
import tunas.core.env as env

if env.ARCH == None:
    _tmp_base_ = os.path.expanduser('~') + '/.tunas/'
    if not os.path.exists(_tmp_base_):
        os.makedirs(_tmp_base_)

    _config_fpath_ = _tmp_base_ + 'config.json'
    if not os.path.exists(_config_fpath_):
    # write default configuration to config.json
        config = {'floatx' : 'float32', 'eps' : '1e-10', 'arch' : 'tensorflow'}
        env.EPS = 1e-7
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
            env.EPS = config['eps']
            env.ARCH = config['arch']
            f.close()

if env.ARCH == 'tensorflow': 
    print('Tensorflow architecture, start initialize session.')
    try:
        import tensorflow as tf
    except:
        raise ImportError("Need Tensorflow as arch dependencies,",
            " and cudnn installed. Please check the configuration.")
    from .tensorflow.expr import *
    from .tensorflow.func import *
    from .tensorflow.math import *
    from .tensorflow.optimization import *
    env.SESSION = tf.Session()
        
elif env.ARCH == 'theano':
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
