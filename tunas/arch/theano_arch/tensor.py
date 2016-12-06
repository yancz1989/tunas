# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-06-19 10:52:49
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-12-04 17:03:06
from __future__ import absolute_import, print_function, division

import numpy as np
import tunas.arch.env as env
import tunas.core.interfaces

import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.nnet.neighbours as nbr
from ..env import get_arch, get_session, get_floatX, get_epsilon

import numpy as np

def arch_name():
  return 'theano'

def release():
  pass

def variable(value, dtype = env.FLOATX, trainable = True, name = None):
  var = theano.shared(value = np.array(value, dtype=dtype), name = name)
  var.trainable = trainable
  return var

def eval(f, feed_dict = None):
  return f.eval(feed_dict)

class Function(object):
  def __init__(self, inputs, outputs):
    self.inputs = inputs
    self.outputs = outputs
    self.func = theano.function(inputs, outputs)

  def __call__(self, datas):
    if len(datas) == 1:
      return self.func(datas[0])
    else:
      return self.func(*datas)

def function(inputs, outputs):
  return Function(inputs, outputs)

def set_value(var, value):
  var.set_value(value)

def get_value(var):
  return var.get_value()

def constant(val, dtype = env.FLOATX, shape = None, name = None):
  return T.TensorType(value = val, dtype = dtype,
    broadcastable = (False, ) * len(shape))(name)

def placeholder(shape = None, dims = None, dtype = env.FLOATX, name = None):
  assert (shape != None or dims != None)
  if dims == None:
    dims = len(shape)
  return T.TensorType(dtype = dtype, broadcastable = (False,) * dims)(name)

def shape(x):
  return x.shape

def dims(x):
  return x.ndim

def size(x):
  return np.prod([dim for dim in x.shape.eval()])

# init variable
def zeros(shape, dtype):
  return variable(np.zeros(shape), dtype)

def ones(shape, dtype):
  return variable(np.ones(shape), dtype)

def ones_like(x):
  return T.ones_like(np.ones(x.shape), x.dtype)

def zeros_like(x):
  return T.zeros_like(x, x.dtype)

def fill(shape, val):
  return variable(np.ones(x.shape) * val, x.dtype)

# variable basic transform
def cast(x, dtype):
  return T.cast(x, dtype)

def linespace(l, r, seps):
  return T.arange(l, r, seps)

def range(start, end, dif):
  # for integer
  return T.arange(start, end, dif)

def concat(vars, dim):
  return T.concatenate(vars, axis = dim)

def rollaxis(x, axis_order):
  return x.dimshuffle(axis_order)

def transpose(x):
  # for 2d matrix
  return T.transpose(x)

def flatten(x):
  return T.flatten(x)

def reshape(x, to):
  return T.reshape(x, to)

def one_hot(indices, depth):
  return T.extra_ops.to_one_hot(indices, depth)

def switch(cond, then_expr, else_expr):
  return T.switch(cond, then_expr, else_expr)

def scan(func, elements, init, opt = None, loops = None):
  return theano.scan(fn = func, sequences = elements,
    outputs_info = init, non_sequences = opt, n_steps = loops)

def map(func, elements):
  return theano.map(fn = func, sequences = elements)

# class Gradient(object):
#   def __init__(self, operator, grad):
#     self.operator = 
    
# WARNING: only matrices are accepted. idxs should be sorted with columns aggragated.
def sparse_tensor(idxs, value, shape):
  indptr = [0]
  indices = []
  col = 0
  for i, idx in zip(range(len(idxs)), idxs):
    if col != idx[1]:
      col = idx[1]
      indptr.append(i)
    indices.append(idx[0])
  return theano.sparse.csc_matrix((value, indices, indptr), shape = shape)

def sparse2dense(sp):
  return theano.sparse.dense_from_sparse(sp)

