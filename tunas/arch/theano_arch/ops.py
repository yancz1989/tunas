# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-06-19 10:52:42
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-12-06 21:30:03
from __future__ import absolute_import, print_function, division

import numpy as np
import tunas.core.interfaces

import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.slinalg as slinalg
from theano.tensor.signal import pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.cuda import dnn

import tunas.arch.env as env
from tunas.arch.universal import _expand, _string_order, dim2d, dim3d, kernel2d, kernel3d, dim_orders, paddings

# math expression
# basic element-wise operator
def add(x, y):
  return T.add(x, y)

def sub(x, y):
  return T.sub(x, y)

def mul(x, y):
  return T.mul(x, y)

def div(x, y, rounding = False):
  # if rounding is True, use floor_div
  if rounding:
    return x // y
  else:
    return x / y

def round(x):
  return T.round(x)

def abs(x):
  return T.abs_(x)

def neg(x):
  return T.neg(x)

def sign(x):
  return T.sgn(x)

def inv(x):
  return T.inv(x)

def square(x):
  return T.sqr(x)

def sqrt(x):
  return T.sqrt(x)

def pow(x, p):
  return T.pow(x, p)

def exp(x):
  return T.exp(x)

def log(x):
  return T.log(x)

def ceil(x):
  return T.ceil(x)

def floor(x):
  return T.floor(x)

def elmw_max(x, y):
  return T.maximum(x, y)

def elmw_min(x, y):
  return T.minimum(x, y)

def sin(x):
  return T.sin(x)

def cos(x):
  return T.cos(x)

def tan(x):
  return T.tan(x)

def asin(x):
  return T.arcsin(x)

def acos(x):
  return T.arccos(x)

def atan(x):
  return T.arctan(x)

# basic linear operator
# transform x to a diag matrix
# e.g. [1,2,3] -> [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
def diag(x):
  return T.nlinalg.diag(x)

# return diag part of tensor
# e.g. [[1, 0, 0], [0, 2, 0], [0, 0, 3]] -> [1, 2, 3]
def diagv(x):
  return T.nlinalg.diag(x)

def trace(x):
  return T.nlinalg.trace(x)

def matmul(x, y):
  return T.dot(x, y)

def batch_matmul(x, y, adj_x = False, adj_y = False):
  # used for tensor, i.e., if x and y are matrix, then use matrix
  # multiplication, else batch_matmul
  # TODO: interfaces with tensorflow backed
  axes = (x.ndim - 1, y.ndim - 2)
  if adj_x == True:
    axes[0] = x.ndim - 2
  if adj_y == True:
    axes[1] = y.ndim - 1
  return T.batched_tensordot(x, y, axes = axes)

def determinant(x):
  # for square matrix
  return T.nlinalg.Det()(x)

def matinv(x):
  # for matrix
  return theano.tensor.nlinalg.MatrixInverse()(x)

def cholesky(x):
  # assert(dims(x) == 2)
  return slinalg.Cholesky()(x)

# fft
def fft(x):
  raise NotImplementedError

def ifft(x):
  raise NotImplementedError

# complex operation
def sum(x, dim = None, keep = False):
  return T.sum(x, axis = dim, keepdims = keep)

def prod(x, dim = None, keep = False):
  return T.prod(x, axis = dim, keepdims = keep)

def max(x, dim = None, keep = False):
  return T.max(x, axis = dim, keepdims = keep)

def min(x, dim = None, keep = False):
  return T.min(x, axis = dim, keepdims = keep)

def argmax(x, dim = None, keep = False):
  return T.argmax(x, axis = dim, keepdims = keep)

def argmin(x, dim = None, keep = False):
  return T.argmin(x, axis = dim, keepdims = keep)

def mean(x, dim = None, keep = False):
  return T.mean(x, axis = dim, keepdims = keep)

def std(x, dim = None, keep = False):
  return T.std(x, axis = dim, keepdims = keep)

def unique(x):
  (val, idx, counts) = T.extra_ops.Unique(False, True, True)(x)
  return (val, idx, counts)

def where(x):
  return T.where(x)

# bool and logical
def eq(x, y):
  return T.eq(x, y)

def neq(x, y):
  return T.neq(x, y)

def lt(x, y):
  return T.lt(x, y)

def le(x, y):
  return T.le(x, y)

def gt(x, y):
  return T.gt(x, y)

def ge(x, y):
  return T.ge(x, y)

def logic_and(x, y):
  return T.bitwise_and(x, y)

def logic_or(x, y):
  return T.bitwise_or(x, y)

def logic_not(x):
  return 1 - x

def logic_xor(x, y):
  return T.bitwise_xor(x, y)

def seed(seed):
  np.random.seed(seed)

# random operation
def randn(shape, _mean = 0, _std = 1.0, dtype = env.FLOATX, seed = np.random.randint(1e5)):
  return RandomStreams(seed = seed).normal(size = shape, avg = _mean, std = _std, dtype = dtype)

def rand(shape, _min = 0, _max = 1, dtype = env.FLOATX, seed = np.random.randint(1e5)):
  return RandomStreams(seed = seed).uniform(size = shape, avg = _mean, std = _std, dtype = dtype)

def binomial(shape, p, dtype = env.FLOATX, seed = np.random.randint(1e5)):
    return RandomStreams(seed = seed).binomial(size = shape, avg = _mean, std = _std, dtype = dtype)

def shuffle(x, seed = np.random.randint(1e5)):
  raise NotImplementedError

# activation function
def relu(x):
  return T.nnet.relu(x)

def softplus(x):
  return T.nnet.softplus(x)

def softmax(x):
  return T.nnet.softmax(x)

def tanh(x):
  return T.tanh(x)

def sigmoid(x):
  return T.nnet.sigmoid(x)

def thresholding(x, thr):
  thr = abs(thr)
  return T.clip(x, -thr, thr)

def clip(x, l, r):
  return T.clip(x, l, r)

def linear(x):
  return x

def batch_normalization(x, mu, sigma, gamma, beta):
  return T.nnet.bn.batch_normalization(x, gamma, beta, mu, sigma)

# convolution neural networks
def _shuffle(x, current, to):
  if current != to:
    x = x.dimshuffle(_string_order(current, to))
  return x

AR = 'th'

def conv2d(x, kernel, strides = (1, 1), padding = 'same', dim_order = 'th'):
  if dim_order not in dim_orders or padding  not in paddings:
    raise Exception('Error dim_order or padding parameter.')
  if padding == 'same':
    padding = 'half'
  x = _shuffle(x, dim2d[dim_order], dim2d[AR])
  kernel = _shuffle(kernel, kernel2d[dim_order], kernel2d[AR])

  if env.get_device() == '':
    conv = nnet.conv2d(x, kernel, subsample = strides, border_mode = padding)
  else:
    conv = dnn.dnn_conv(x, kernel, border_mode = padding, subsample = strides)

  if padding == 'half':
    shape = kernel.shape.eval()
    if shape[2] % 2 == 0:
      conv = conv[:, :, 1:, :]
    if shape[3] % 2 == 0:
      conv = conv[:, :, :, 1:]
  return _shuffle(conv, dim2d[AR], dim2d[dim_order])

def conv3d(x, kernel, strides = (1, 1, 1), padding = 'same', dim_order = 'th'):
  if dim_order not in dim_orders or padding  not in paddings:
    raise Exception('Error dim_order or padding parameter.')
  if env.get_device() == '':
    raise NotImplementedError
  x = _shuffle(x, dim3d[dim_order], dim3d[AR])
  kernel = _shuffle(kernel, kernel3d[dim_order], kernel3d[AR])
  if padding == 'same':
    padding = 'half'
  conv = dnn.dnn_conv3d(x, kernel, border_mode = padding, subsample = strides)
  if padding == 'half':
    shp = kernel.shape.eval()
    if shp[2] % 2 == 0:
      conv = conv[:, :, 1:, :, :]
    if shp[3] % 2 == 0:
      conv = conv[:, :, :, 1:, :]
    if shp[4] % 2 == 0:
      conv = conv[:, :, :, :, 1:]
  return _shuffle(conv, dim3d[AR], dim3d[dim_order])

def pool2d(x, pool_size, strides, padding, dim_order, mode):
  if dim_order not in dim_orders or padding  not in paddings:
    raise Exception('Error dim_order or padding parameter.')
  x = _shuffle(x, dim2d[dim_order], dim2d[AR])
  if padding == 'same':
    shp = x.shape.eval()
    pad = tuple([(pool_size[i] - 2) if pool_size[i] % 2 == 1 else (pool_size[i] - 1) for i in range(len(pool_size))])
    expected = [shp[i + 2] + strides[i] - 1 // strides[i] for i in range(len(strides))]
  else:
    pad = (0, 0)
  if env.get_device() == '':
    pool_out = pool.pool_2d(x, ds = pool_size, st = strides, ignore_border = True, padding = pad, mode = mode)
  else:
    pool_out = dnn.dnn_pool(x, pool_size, stride = strides, mode = mode, pad = pad)
  if padding == 'same':
    pool_out = pool_out[:, :, : expected[0], : expected[1]]
  return _shuffle(pool_out, dim2d[AR], dim2d[dim_order])

# pool_size is the pool width of (time, height, width), the same with strides
def pool3d(x, pool_size, strides, padding, dim_order, mode):
  if dim_order not in dim_orders or padding  not in paddings:
    raise Exception('Error dim_order or padding parameter.')
  x = _shuffle(x, dim3d[dim_order], dim3d[AR])
  if padding == 'same':
    shp = x.shape.eval()
    pad = tuple([(s - 2) if s % 2 == 1 else (s - 1) for s in pool_size])
    expected = [shp[i + 2] + strides[i] - 1 // strides[i] for i in range(len(strides))]
  else:
    pad = (0, 0, 0)
  if env.get_device() == '':
    raise NotImplementedError
  else:
    output = dnn.dnn_pool(x, pool_size, stride = strides, mode = mode, pad = pad)
  if padding == 'same':
    output = output[:, :, : expected[0], : expected[1], : expected[2]]
  return _shuffle(output, dim3d[AR], dim3d[dim_order])

def max_pool2d(x, pool_size, strides, padding = 'same', with_arg = False, dim_order = 'th'):
  if with_arg:
    raise NotImplementedError('Argmax option is not enabled in Theano arch.')
  return pool2d(x, pool_size, strides, padding, dim_order, 'max')

def max_pool3d(x, pool_size, strides, padding = 'same', dim_order = 'th'):
  return pool3d(x, pool_size, strides, padding, dim_order, 'max')

def avg_pool2d(x, pool_size, strides, padding = 'same', dim_order = 'th'):
  return pool2d(x, pool_size, strides, padding, dim_order, 'average_exc_pad')

def avg_pool3d(x, pool_size, strides, padding = 'same', dim_order = 'th'):
  return pool3d(x, pool_size, strides, padding, dim_order, 'average_exc_pad')

def window_slides(var, ksize, strides):
  return nbr.images2neibs(var, ksize, strides, mode = 'valid')

def dropout(x, p, seed = np.random.randint(1e5)):
  rng = RandomStreams(seed=seed)
  return x * rng.binomial(x.shape, p=retain_prob, dtype=x.dtype) / (1. - p)

def to_one_hot(idx, labels):
  return T.extra_ops.to_one_hot(T.cast(T.flatten(idx), 'int32'), nb_class=labels)

def l2_normalize(x, dim):
  return x / (T.sqrt(T.sum(T.square(x), axis=dim, keepdims=True)) + env.EPS)

# common objectives
def mse(gt, pred):
  # mean square error
  return mean(square(gt - pred))

def mae(gt, pred):
  # mean absolute error
  return mean(abs(gt - pred))

def msle(gt, pred):
  # mean square log error
  return mean(square(log(pred + 1.0) - log(gt + 1.0)))

def sqr_hinge(gt, pred):
  return mean(square(max(1.0 - mul(pred, gt), 0)))

def hinge(gt, pred):
  return mean(max(1.0 - mul(pred, gt)))

def categorical_crossentropy(gt, pred, prob = False):
  if not prob:
    pred = T.nnet.softmax(pred)
  else:
    pred /= pred.sum(axis=-1, keepdims = True)
  return T.nnet.categorical_crossentropy(T.clip(pred, env.EPS, 1.0 - env.EPS), gt)

def sparse_categorical_crossentropy(gt, pred, prob = True):
  return categorical_crossentropy(pred, reshape(T.extra_ops.to_one_hot(
    T.cast(T.flatten(gt), 'int32'), nb_class=pred.shape[-1]), shape(pred)), prob)

def binary_crossentropy(gt, pred, prob = False):
  if not prob:
    pred = T.nnet.sigmoid(pred)
  return T.nnet.binary_crossentropy(T.clip(pred, env.EPS, 1.0 - env.EPS), gt)

def cosine_proximity(gt, pred):
  return -mean(l2_normalize(gt, dim = 0) * l2_normalize(pred, dim = 0))

# optimization
def grad(x, y):
  # for function y = f(x), input should be function variable y and parameter
  # variable x
  return T.grad(y, x)


class Gradient(object):
  def __init__(self, operator, grad):
    pass
    # self.operator = 

# Here the optimizer is a class, which has method minimize.
# For Theano, please implement its class and minimize method.

# Here no stochastic batch generator is implemented, as you will need to
# shuffle and write generater by your self.
class GradientDescentOptimizer():
  def __init__(self):
    pass

  def minimize(self):
    pass

  def compute_gradients(self):
    pass

  def apply_gradients(self):
    pass

def gd():
  return GradientDescentOptimizer()

class MomentumOptimizer():
  def __init__(self):
    pass

  def minimize(self):
    pass

  def compute_gradients(self):
    pass

  def apply_gradients(self):
    pass

def momentum():
  return MomentumOptimizer()

# TODO: implement adam based on http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
class RMSPropOptimizer():
  def __init__(self):
    pass

  def minimize(self):
    pass

  def compute_gradients(self):
    pass

  def apply_gradients(self):
    pass

def rmsprop():
  return RMSPropOptimizer()

# TODO: implement adam based on http://www.magicbroom.info/Papers/DuchiHaSi10.pdf
class AdagradOptimizer():
  def __init__(self):
    pass

  def minimize(self):
    pass

  def compute_gradients(self):
    pass

  def apply_gradients(self):
    pass

def adagrad():
  return AdagradOptimizer()

# TODO: implement adam based on https://arxiv.org/pdf/1212.5701v1.pdf
class AdadeltaOptimizer():
  def __init__(self):
    pass

  def minimize(self):
    pass

  def compute_gradients(self):
    pass

  def apply_gradients(self):
    pass

def adadelta():
  return AdadeltaOptimizer()

# TODO: implement adam based on https://arxiv.org/pdf/1412.6980.pdf
class AdamOptimizer():
  def __init__(self):
    pass

  def minimize(self):
    pass

  def compute_gradients(self):
    pass

  def apply_gradients(self):
    pass

def adam():
  return AdamOptimizer()

# TODO: implement adamax based on  https://arxiv.org/pdf/1412.6980
class AdamaxOptimizer():
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
