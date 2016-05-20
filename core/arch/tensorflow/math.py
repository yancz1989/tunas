# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-05 21:00:55
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-20 20:35:57

# This file implement interfaces for math operations, including linear and non-linear operation,
# random flow, and basic convolution and recurrent operation.

# basic element-wise operator
import tensorflow as tf
import numpy as np
from .expr import *
import tunas.core.env as env

def add(x, y, name = None):
    return tf.add(x, y, name)

def sub(x, y, name = None):
    return tf.sub(x, y, name)

def mul(x, y, name = None):
    return tf.mul(x, y, name)

def div(x, y, rounding = False, name = None):
    # if rounding is True, use floor_div
    if rounding:
        return tf.floordiv(x, y, name)
    else:
        return tf.div(x, y, name)

def round(x, name = None):
    return tf.round(x, name)

def abs(x, name = None):
    return tf.abs(x, name)

def neg(x, name = None):
    return tf.neg(x, name)

def sign(x, name = None):
    return tf.sign(x, name)

def inv(x, name = None):
    return tf.inv(x, name)

def square(x, name = None):
    return tf.square(x, name)

def sqrt(x, name = None):
    return tf.sqrt(x, name)

def pow(x, p, name = None):
    return tf.pow(x, p, name)

def exp(x, name = None):
    return tf.exp(x, name)

def log(x, name = None):
    return tf.log(x, name)

def ceil(x, name = None):
    return tf.ceil(x, name)

def floor(x, name = None):
    return 

def elmw_max(x, y, name = None):
    return tf.max(x, y, name)

def elmw_min(x, y, name = None):
    return tf.min(x, y, name)

def sin(x, name = None):
    return tf.sin(x, name)

def cos(x, name = None):
    return tf.cos(x, name)

# basic linear operator
def diag(x, name = None):
    return tf.diag(x, name)

def diagv(x, name = None):
    return tf.diag_part(x, name)

def trace(x, name = None):
    return tf.trace(x, name)

def transpose(x, name = None):
    # for 2d matrix
    # assert dims(x) == 2
    return tf.transpose(x, name)

def roll(x, raxs, name = None):
    return tf.transpose(x, raxs, name)

def flatten(x, name = None):
    return tf.reshape(x, [-1], name = name)

def reshape(x, to, name = None):
    return tf.reshape(x, to, name)

def matmul(x, y, name = None):
    # used for tensor, i.e., if x and y are matrix, then use matrix
    # multiplication, else batch_matmul
    if dims(x) == 2 and dims(y) == 2:
        r = tf.matmul(x, y)
    else:
        r = tf.batch_matmul(x, y, adj_x = None, adj_y = None)

def pad(x, pad):
    # pad should be compatible for dimensions of x, in form of [[1,1], [2,2]]
    # assert dims(x) < 3
    # assert dims(x) == len(pad):
    return tf.pad(x, pad)


def determinant(x, name = None):
    # for square matrix
    # assert dims(x) == 2
    return tf.determinant(x, name)


def matinv(x, name = None):
    # for matrix
    # assert dims(x) == 2
    return tf.matrix_inverse(x, name)


def cholesky(x, name = None):
    # assert(dims(x) == 2)
    return tf.cholesky(x, name)

# fft
def fft(x, name = None):
    x = x.as_dtype('complex64')
    if dims(x) == 1:
        ret = tf.fft(x, name = name)
    elif dims(x) == 2:
        ret = tf.fft2d(x, name = name)
    elif dims(x) == 3:
        ret = tf.fft3d(x, name = name)
    return ret

def ifft(x, name = None):
    x = x.as_dtype('complex64')
    if dims(x) == 1:
        ret = tf.fft(x, name = name)
    elif dims(x) == 2:
        ret = tf.fft2d(x, name = name)
    elif dims(x) == 3:
        ret = tf.fft3d(x, name = name)
    return ret


# complex operation
def sum(x, dim = None, keep = False, name = None):
    return tf.reduce_sum(x, dim, keep_dims = keep, name = name)

def prod(x, dim = None, name = None):
    return tf.reduce_prod(x, dim, name = name)

def max(x, dim = None, name = None):
    return tf.reduce_max(x, dim, name = name)

def min(x, dim = None, name = None):
    return tf.reduce_min(x, dim, name = name)

def argmax(x, dim = None, name = None):
    return tf.argmax(x, dim, name = name)

def argmin(x, dim = None, name = None):
    return tf.argmin(x, dim, name = name)

def mean(x, dim = None, name = None):
    return tf.reduce_mean(x, dim, name = name)

def std(x, dim = None, name = None):
    return tf.sqrt(tf.reduce_mean(tf.square(x - tf.mean(x, dim), dim)), name = name)

def unique(x, name = None):
    return tf.unique(x, name = name)

def where(x, name = None):
    return tf.where(x, name = name)

# bool and logical
def equal(x, y):
    return tf.equal(x, y)

def less(x, y):
    return tf.less(x, y)

def less_equal(x, y):
    return tf.less_equal(x, y)

def greater(x, y):
    return tf.greater(x, y)

def greater_equal(x, y):
    return tf.greater_equal(x, y)

def logic_and(x, y):
    return tf.logical_and(x, y)

def logic_or(x, y):
    return tf.logical_or(x, y)

def logic_not(x, y):
    return tf.logical_not(x, y)

def logic_xor(x, y):
    return tf.logical_xor(x, y)

def switch(condition, then_expr, else_expr):
    return tf.cond(condition, then_expr, else_expr)

# random operation
def srand(s):
    tf.set_random_seed(s)

def randn(shape, _mean = 0, _std = 1.0, thr = None, dtype = None, seed = None, name = None):
    if dtype == None:
        dtype = env.FLOATX
    if thr == None:
        var = tf.random_norm(shape, _mean, _std, dtype = dtype, seed = seed, name = name)
    else:
        # if thr is not None, then use truncated normal, for 2 sigma
        var = tf.truncated_normal(shape, _mean, _std, dtype, seed, name)
    return 

def rand(shape, _min = 0, _max = 1, dtype = None, seed = None, name = None):
    if dtype == None:
        dtype = env.FLOATX
    return tf.random_uniform(shape, _min, _max, dtype = dtype, seed = seed, name = name)

def shuffle(x, seed = None, name = None):
    return tf.shuffle(x, seed, name)
