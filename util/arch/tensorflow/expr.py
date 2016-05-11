# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-05 20:59:31
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-11 16:57:51

# This file implement expression module for tunas, including variable,
# placeholder, function and their base support interfaces.
import tensorflow as tf
import numpy as np

global _EPS_
global _FLOATX_
global _ARCH_
global _SESSION_

def get_session():
    global _SESSION_
    return _SESSION_

def variable(value, dtype, name):
    var = tf.Variable(np.asarray(value, dtype = dtype), name = name)
    get_session().run(v.initializer)
    return var

def constant(value, shape, dtype = None, name = None):
    if dtype == None:
        dtype = _FLOATX_
    return tf.constant(value, dtype, shape, name)

def placeholder(shape, dims, dtype, name):
    assert (shape != None or dims != None)
    if shape == None:
        shape = [None for _ in range(dims)]
    return tf.placeholder(dtype = dtype, shape = shape, name = name)

def shape(x):
    return tf.shape(x)

def dims(x):
    return len(tf.shape(x))

def size(x):
    return np.prod([dim._value for dim in x.get_shape(x)])

def rank(x):
    return tf.rank(x)

# init variable
def zeros(shape, dtype, name):
    return variable(np.zeros(shape), dtype, name)

def ones(shape, dtype, name):
    return variable(np.ones(shape), dtype, name)

def ones_like(x, name):
    return tf.ones_like(x, name=name)

def zeros_like(x, name):
    return tf.zeros_like(x, name=name)

def fill(shape, val):
    return tf.fill(shape, val)

# variable basic transform
def cast(x, dtype):
    return tf.cast(x, dtype)

def linespace(l, r, seps):
    return tf.linespace(l, r, seps)

def range(start, end, dif):
    # for integer
    return tf.range(start, end, dif)


