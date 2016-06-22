# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-06-19 09:23:30
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-06-22 21:59:58

# math expression
# basic element-wise operator
# please note operators are overrided including:
# neg(-), abs(), invert(~), add(+), sub(-), mul(*), div(/), floordiv(//), mod(%)
# pow(**), and(&), or(|), xor(^), lt(<), gt(>), le(<=), ge(>=)
from __future__ import absolute_import, print_function, division  

import tensorflow as tf
import numpy as np
import tunas.core.env as env
import tunas.core.interfaces
from tunas.core.arch.universal import _expand, _string_order, dim2d, dim3d, kernel2d, kernel3d, dim_orders, paddings

def add(x, y):
  return tf.add(x, y)

def sub(x, y):
  return tf.sub(x, y)

def mul(x, y):
  return tf.mul(x, y)

def div(x, y, rounding = False):
  # if rounding is True, use floor_div
  if rounding:
    return tf.floordiv(x, y)
  else:
    return tf.truediv(x, y)

def round(x):
  return tf.round(x)

def abs(x):
  return tf.abs(x)

def neg(x):
  return tf.neg(x)

def sign(x):
  return tf.sign(x)

def inv(x):
  return tf.inv(x)

def square(x):
  return tf.square(x)

def sqrt(x):
  return tf.sqrt(x)

def pow(x, p):
  return tf.pow(x, p)

def exp(x):
  return tf.exp(x)

def log(x):
  return tf.log(x)

def ceil(x):
  return tf.ceil(x)

def floor(x):
  return tf.floor(x)

def elmw_max(x, y):
  return tf.maximum(x, y)

def elmw_min(x, y):
  return tf.minimum(x, y)

def sin(x):
  return tf.sin(x)

def cos(x):
  return tf.cos(x)

def tan(x):
  return tf.tan(x)

def asin(x):
  return tf.asin(x)

def acos(x):
  return tf.acos(x)

def atan(x):
  return tf.atan(x)

# basic linear operator
def diag(x):
  return tf.diag(x)

def diagv(x):
  return tf.diag_part(x)

def trace(x):
  return tf.trace(x)

# matrix multiply
def matmul(x, y):
  # used for tensor, i.e., if x and y are matrix, then use matrix
  # multiplication, else batch_matmul
  return tf.matmul(x, y)

# for batch_maumul, e.g. x shape [..., x_r, x_c], y shape [..., y_r, y_c]
# for each element in output tensor z whose shape is determined by adj_x, adj_y,
# by default z[..., i, j] = \sum_{k}x[..., i, k] * y[..., k, j], required x_c = y_r
# if only adj_x is True, z[..., i, j] = \sum_{k}x[..., k, i] * y[..., k, j], required x_r = y_r
# if only adj_y is True, z[..., i, j] = \sum_{k}x[..., i, k] * y[..., j, k], require x_c = y_c
# if both adj_x and adj_y is True, z[...,i,j] = \sum_{k}x[...,k,i]*y[...,j,k], require x_r = y_c
def batch_matmul(x, y, adj_x = False, adj_y = False):
  return tf.batch_matmul(x, y, adj_x = adj_x, adj_y = adj_y)

def determinant(x):
  # for square matri
  # assert dims(x) == 2
  return tf.matrix_determinant(x)

def matinv(x):
  # for matrix
  return tf.matrix_inverse(x)

def cholesky(x):
  return tf.cholesky(x)

# fft
def fft(x):
  x = x.as_dtype('complex64')
  if dims(x) == 1:
    ret = tf.fft(x)
  elif dims(x) == 2:
    ret = tf.fft2d(x)
  elif dims(x) == 3:
    ret = tf.fft3d(x)
  return ret

def ifft(x):
  x = x.as_dtype('complex64')
  if dims(x) == 1:
    ret = tf.fft(x)
  elif dims(x) == 2:
    ret = tf.fft2d(x)
  elif dims(x) == 3:
    ret = tf.fft3d(x)
  return ret

# complex operation
def sum(x, dim = None, keep = False):
  return tf.reduce_sum(x, reduction_indices = dim, keep_dims = keep)

def prod(x, dim = None, keep = False):
  return tf.reduce_prod(x, reduction_indices = dim, keep_dims = keep)

def max(x, dim = None, keep = False):
  return tf.reduce_max(x, reduction_indices = dim, keep_dims = keep)

def min(x, dim = None, keep = False):
  return tf.reduce_min(x, reduction_indices = dim, keep_dims = keep)

def argmax(x, dim = None, keep = False):
  if dim == None:
    return tf.argmax(tf.reshape(x, [-1]), dimension = 0)
  else:
    return tf.argmax(x, dimension = dim)

def argmin(x, dim = None, keep = False):
  if dim == None:
    return tf.argmin(tf.reshape(x, [-1]), dimension = 0)
  else:
    return tf.argmin(x, dimension = dim)

def mean(x, dim = None, keep = False):
  return tf.reduce_mean(x, reduction_indices = dim, keep_dims = keep)

def std(x, dim = None, keep = False):
  return tf.sqrt(tf.reduce_mean(tf.square(x - tf.reduce_mean(
    x, reduction_indices=dim, keep_dims=True)), reduction_indices = dim, keep_dims = keep))

def unique(x):
  val, idx, counts = tf.unique_with_counts(tf.reshape(x, [-1]))
  return (val, idx, counts)

# bool and logical
def eq(x, y):
  return tf.equal(x, y)

def neq(x, y):
  return tf.not_equal(x, y)

def lt(x, y):
  return tf.less(x, y)

def le(x, y):
  return tf.less_equal(x, y)

def gt(x, y):
  return tf.greater(x, y)

def ge(x, y):
  return tf.greater_equal(x, y)

def logic_and(x, y):
  return tf.logical_and(x, y)

def logic_or(x, y):
  return tf.logical_or(x, y)

def logic_not(x):
  return tf.logical_not(x)

def logic_xor(x, y):
  return tf.logical_xor(x, y)

# random operation
def randn(shape, _mean = 0, _std = 1.0, dtype = env.FLOATX, seed = np.random.randint(1e5)):
  return tf.random_norm(shape, _mean, _std, dtype = dtype, seed = seed)

def rand(shape, _min = 0, _max = 1, dtype = env.FLOATX, seed = np.random.randint(1e5)):
  return tf.random_uniform(shape, _min, _max, dtype = dtype, seed = seed)

def binomial(shape, p, dtype = env.FLOATX, seed = np.random.randint(1e5)):
  return tf.select(tf.random_uniform(shape, dtype = dtype, seed=seed) <= p,
                     tf.ones(shape), tf.zeros(shape))

def shuffle(x, seed = None):
  return tf.shuffle(x, seed)

# activation function
def relu(x):
  return tf.nn.relu(x)

def softplus(x):
  return tf.nn.softplus(x)

def softmax(x):
  return tf.nn.softmax(x)

def tanh(x):
  return tf.nn.tanh(x)

def sigmoid(x):
  return tf.nn.sigmoid(x)

def thresholding(x, thr):
  thr = abs(thr)
  return tf.clip_by_value(x, -thr, thr)

def clip(x, l, r):
  return tf.clip_by_value(x, l, r)

def linear(x):
  return x

def batch_normalization(x, mu, sigma, gamma, beta):
  return tf.nn.batch_normalization(x, mu, sigma, beta, gamma, variance_epsilon = env.EPS)

# convolution neural networks
def _shuffle(x, current, to):
  if current != to:
    x = tf.transpose(x, _string_order(current, to))
  return x

AR = 'tf'

# border choose from 'valid' or 'same'
# dim_order choose from 'NHWC' or 'NCHW'
# If variant dim_order is not support, we use _transpose_on to transpose
# at the beginning and the end of program to keep the input and output
# in the same dimension order. It is suggested that in Tensorflow, we use
# 'NHWC' while in theano use 'NCHW' as data format for its native support.
# For 'NHWC':
#   dim_order: nhwc, i.e. batch_size, height, width, channels
#   kernel_order: hwcd, i.e. kernel height, kernel width, channels, depth
# For 'NCHW':
#   dim_order: nchw, i.e. batch_size, channels, height, width
#   kernel_order: dchw, i.e. depth, channels, kernel height, kernel width
# For 3d convolution, we adopt its usage in video recognition, as 't' stand
# for time frame.

def conv2d(x, kernel, strides, padding = 'same', dim_order = 'tf'):
  if dim_order not in dim_orders or padding  not in paddings:
    raise Exception('Error dim_order or padding parameter.')
  print(dim2d[dim_order])
  return tf.nn.conv2d(x, kernel, _expand(strides),
    padding = padding.upper(), data_format = dim2d[dim_order])

def conv3d(x, kernel, strides, padding = 'same', dim_order = 'tf'):
  if dim_order not in dim_orders or padding  not in paddings:
    raise Exception('Error dim_order or padding parameter.')
  return _shuffle(tf.nn.conv3d(_shuffle(x, dim3d[dim_order], dim3d[AR]),
    _shuffle(kernel, kernel3d[dim_order], kernel3d[AR]),
    _expand(strides), padding.upper()), dim3d[AR], dim3d[dim_order])

# if with_arg is True, output is combined with output and argmax,
# the latter is flattened indices of argmax.
def max_pool2d(x, pool_size, strides, padding = 'same', with_arg = False, dim_order = 'nhwc'):
  if dim_order not in dim_orders or padding  not in paddings:
    raise Exception('Error dim_order or padding parameter.')
  if with_arg == False:
    return tf.nn.max_pool(x, _expand(pool_size), _expand(strides),
      padding = padding.upper(), data_format = dim2d[dim_order])
  else:
    return _shuffle(tf.nn.max_pool_with_argmax(_shuffle(x, dim2d[dim_order], dim2d[AR]),
      _expand(pool_size), _expand(strides), padding = padding.upper()), dim2d[AR], dim2d[dim_order])

def max_pool3d(x, pool_size, strides, padding = 'same', dim_order = 'ndhwc'):
  if dim_order not in dim_orders or padding  not in paddings:
    raise Exception('Error dim_order or padding parameter.')
  return _shuffle(tf.nn.max_pool3d(_shuffle(x, dim3d[dim_order], dim3d[AR]),
    _expand(pool_size), _expand(strides), padding.upper()), dim3d[AR], dim3d[dim_order])

def avg_pool2d(x, pool_size, strides, padding = 'same', dim_order = 'ndhwc'):
  if dim_order not in dim_orders or padding  not in paddings:
    raise Exception('Error dim_order or padding parameter.')
  return tf.nn.avg_pool(x, _expand(pool_size), _expand(strides),
    padding.upper(), data_format = dim2d[dim_order])

def avg_pool3d(x, pool_size, strides, padding = 'same', dim_order = 'ndhwc'):
  if dim_order not in dim_orders or padding  not in paddings:
    raise Exception('Error dim_order or padding parameter.')
  return _shuffle(tf.nn.avg_pool3d(_shuffle(x, dim3d[dim_order], dim3d[AR]),
    _expand(pool_size), _expand(strides), padding.upper()), dim3d[AR], dim3d[dim_order])

# from tensor to window slides
def window_slides(var, ksize, strides):
  return tf.extract_image_patches(var, 'VALID', _expand(ksize),
    _expand(strides), rates = (1, 1, 1, 1))

def dropout(x, p, seed):
  pr = 1.0 - p
  return tf.nn.dropout(x, pr, seed)

def to_one_hot(idx, labels):
  return tf.one_hot(tf.cast(idx, dtype='int32'), labels)

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
  if prob:
    dim = len(pred.get_shape()) - 1
    pred = tf.log(tf.clip_by_value(pred / tf.reduce_sum(pred,
      reduction_indices = dim, keep_dims = True), env.EPS, 1 - env.EPS))
    loss = -tf.reduce_sum(gt * pred, reduction_indices = dim)
  else:
    loss = tf.nn.softmax_cross_entropy_with_logits(pred, gt)
  return loss

def binary_crossentropy(gt, pred, prob = False):
  if prob:
    pred = tf.clip_by_value(pred, env.EPS, 1 - env.EPS)
    pred = tf.log(pred / (1 - pred))
  return tf.nn.sigmoid_cross_entropy_with_logits(pred, gt)

def sparse_softmax_crossentropy(gt, pred, prob = True):
  if prob:
    pred = tf.log(tf.clip_by_value(pred, env.EPS, 1 - env.EPS))
  return tf.sparse_softmax_cross_entropy_with_logits(pred, gt)

def cosine_proximity(gt, pred):
  return -mean(tf.nn.l2_normalize(gt, dim = 0) * tf.nn.l2_normalize(pred, dim = 0))

# optimization
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
            eps = env.EPS, use_locking = False):
  return tf.train.RMSPropOptimizer()

def adagrad(learning_rate, initial_accumulator_value = 0.1, use_locking = False):
  return tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value, use_locking)

def adadelta(learning_rate = 0.001, rho = 0.95, eps = env.EPS, use_lock = False):
  return tf.train.AdadeltaOptimizer(learning_rate, rho, eps, use_lock)

def adam(learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999,
    eps = env.EPS, use_locking = False):
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

