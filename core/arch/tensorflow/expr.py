# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-05 20:59:31
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-20 20:35:42

# This file implement expression module for tunas, including variable,
# placeholder, function and their base support interfaces.
import tensorflow as tf
import numpy as np
import tunas.core.env as env

def get_session():
    return env.SESSION

def variable(value, dtype = env.FLOATX, trainable = True, name = None):
    var = tf.Variable(np.asarray(value, dtype = dtype), trainable = trainable, name = name)
    return var

def init_variable(list_vars):
    return tf.initialize_variables(list_vars)

def eval(f, feed_dict = None):
    return env.SESSION.run(f, feed_dict = feed_dict)

def set_value(var, value):
    return var.assign(value)

def get_value(var):
    return env.SESSION.run(var)

def constant(value, dtype = env.FLOATX, shape = None, name = None):
    if dtype == None:
        dtype = env.FLOATX
    return tf.constant(value, dtype, shape, name)

def placeholder(shape = None, dims = None, dtype = env.FLOATX, name = None):
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


