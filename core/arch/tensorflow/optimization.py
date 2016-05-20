# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-05 20:58:25
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-20 20:36:03

# This file implement interface for optimization used for neural networks training.
from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf
import tunas.core.env as env
import tunas.core.interfaces

def grad(x, y):
    # for function y = f(x), input should be function variable y and parameter
    # variable x
    return tf.gradients(y, x)

# Here the optimizer is a class, which has method minimize.
# For Theano, please implement its class and minimize method.

# Here no stochastic batch generator is implemented, as you will need to
# shuffle and write generater by your self.
def gd(learning_rate = 0.1, use_locking = False):
    return tf.train.GradientDescentOptimizer(learning_rate, use_locking)

def momentum(learning_rate = 0.1, momentum_rate = 0.9,
        use_locking = False):
    return tf.train.MomentumOptimizer(learning_rate,
        momentum_rate, use_locking)

def rmsprop(objective, learning_rate, decay = 0.9, momentum = 0.0,
        eps = None, use_locking = False):
    return tf.train.RMSPropOptimizer()

def adagrad(learning_rate, initial_accumulator_value = 0.1, 
        use_locking = False):
    return tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value, use_locking)

def adadelta(learning_rate = 0.001, rho = 0.95, eps = None,
        use_lock = False):
    if eps == None:
        eps = env.EPS
    return tf.train.AdadeltaOptimizer(learning_rate, rho, eps, use_lock)

def adam(learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999,
        eps = None, use_locking = False):
    if eps == None:
        eps = env.EPS
    return tf.train.AdamOptimizer(learning_rate, beta1, beta1,
        eps, use_locking)

# TODO: implement adamax from https://arxiv.org/pdf/1412.6980

class AdamaxOptimizer(tunas.core.interfaces.Optimizer):
    def __init__(self):
        pass

    def minimize(self):
        pass

    def compute_gradients(self):
        pass

    def apply_gradients(self):
        pass

def adamax():
    return AdamaxOptimizer()