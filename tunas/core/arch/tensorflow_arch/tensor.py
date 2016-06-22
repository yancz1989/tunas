# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-06-19 09:23:16
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-06-20 00:24:49
from __future__ import absolute_import, print_function, division  

import tensorflow as tf
import numpy as np
import tunas.core.env as env
import tunas.core.interfaces

def arch_name():
  return 'tensorflow'

def _get_session():
  return env.SESSION

def release():
  env.SESSION.close()

def variable(value, dtype = env.FLOATX, trainable = True, name = None):
  var = tf.Variable(np.array(value, dtype = dtype), trainable = trainable, name = name)
  _get_session().run(var.initializer)
  return var

def eval(f, feed_dict = None):
  return _get_session().run(f, feed_dict = feed_dict)

class Function(object):
  def __init__(self, inputs, outputs):
    self.inputs = inputs
    self.outputs = outputs

  def __call__(self, datas):
    return eval(self.outputs, feed_dict = {input.name : data
      for (input, data) in zip(self.inputs, datas)})

def function(inputs, outputs):
  return Function(inputs, outputs)

def set_value(var, value):
  _get_session().run(var.assign(value))

def get_value(var):
  return _get_session().run(var)

def constant(value, dtype = env.FLOATX, shape = None, name = None):
  return tf.constant(value, dtype, shape, name = name)

def placeholder(shape = None, dims = None, dtype = env.FLOATX, name = None):
  assert (shape != None or dims != None)
  if shape == None:
    shape = [None for _ in range(dims)]
  return tf.placeholder(dtype = dtype, shape = shape, name = name)

def shape(x):
  return x.get_shape()

def dims(x):
  return x.get_shape()._dims

def size(x):
  return np.prod([dim._value for dim in x.get_shape(x)])

# init variable
def zeros(shape, dtype = env.FLOATX):
  return variable(np.zeros(shape), dtype)

def ones(shape, dtype = env.FLOATX):
  return variable(np.ones(shape), dtype)

def ones_like(x):
  return tf.ones_like(x, dtype = env.FLOATX)

def zeros_like(x):
  return tf.zeros_like(x, dtype = env.FLOATX)

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

def concat(vars, dim):
  return tf.concat(dim, vars)

def transpose(x):
  # for 2d matrix
  return tf.transpose(x)

def rollaxis(x, axis_order):
    return tf.transpose(x, axis_order)

def flatten(x):
  return tf.reshape(x, [-1])

def reshape(x, to):
  return tf.reshape(x, to)

# useful when transform label to matrix
# e.g. indices = [0, 1, -1, 2], depth = 3 --> [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
def one_hot(indices, depth):
  return tf.one_hot(indices, depth)

# control flow
def switch(condition, then_expr, else_expr):
  return tf.cond(condition, then_expr, else_expr)

def scan(func, elements, init, opt = None, loops = None):
  return tf.scan(func, elements, initializer = init, n_steps = loops)

def map(func, elements):
  return tf.map(func, elements)

# sparse operator, idxs of format [[0, 1], [1, 2], [2, 3]]
def sparse_tensor(idxs, value, shape):
  return tf.SparseTensor(idxs, value, shape)

def sparse2dense(sp):
  return tf.sparse_tensor_to_dense(sp)

