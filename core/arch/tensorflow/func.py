# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-05 21:01:09
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-20 20:35:53

# This file implement interfaces of functions used for neural networks, including function like
# sigmoid, softmax, normalize, mean, max, min, etc.

import tensorflow as tf
import numpy as np
from .expr import *
from .math import *
import tunas.core.env as env

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

# convolution neural networks
def pool2d(x, pool_size, strides = (1, 1), method = 'max', border = 'valid', dim_order = 'nhwc'):
    # method choose from 'max' or 'avg'
    # border choose from 'valid' or 'same'
    # dim_order choose from 'NHWC' or 'NCHW'
    pool_size = (1, ) + pool_size + (1, )
    strides = (1, ) + strides + (1, )
    # assert method in {'max', 'avg'} and border in {'valid', 'same'} and dim_order in {'nhwc', 'nchw'}
    if method == 'max':
        pool = tf.nn.max_pool(x, pool_size, strides, 
            padding = border.upper(), data_format = dim_order.upper())
    else:
        pool = tf.nn.avg_pool(x, pool_size, strides, 
            padding = border.upper(), data_format = dim_order.upper())

def conv2d(x, kernel, strides, border = 'valid', dim_order = 'nhwc'):
    assert border in {'valid', 'same'} and dim_order in {'nhwc', 'nchw'}
    strides = (1, ) + strides + (1, )
    return tf.nn.conv2d(x, kernel, strides, padding = border.upper(), data_format = dim_order)

def dropout(x, p, seed):
    pr = 1.0 - p
    return tf.nn.dropout(x, pr, seed)

def l2_normalize(x, dim):
    return tf.nn.l2_normalize(x, dim = dim)

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

def squred_hinge(gt, pred):
    return mean(sqare(max(1.0 - mul(pred, gt), 0)))

def hinge(gt, pred):
    return mean(max(1.0 - mul(pred, gt)))

def categorical_crossentropy(gt, pred, prob = True):
    if prob:
        loss = -(sum(gt * log(clip(pred / sum(pred, dim = dims(pred) - 1, keep = True),
            env.EPS, 1.0 - env.EPS)), dims(pred) - 1))
    else:
        loss = tf.nn.softmax_cross_entropy_with_logits(pred, gt)
    return loss

def binary_crossentropy(gt, pred):
    predn = clip(pred, env.EPS, 1.0 - env.EPS)
    return mean(tf.nn.sigmoid_cross_entropy_with_logits(
        log(predn / (1 - predn)), gt))

def cosine_proximity(gt, pred):
    return -mean(l2_normalize(gt) * l2_normalize(pred))


