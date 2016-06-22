# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-06-20 11:00:29
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-06-20 11:00:31

import numpy as np

import tunas.core.arch.theano_mod.ops as ops
reload(ops)
from tunas.core.arch.theano_mod.ops import *

np.random.seed(2012310818)
x = theano.shared(np.random.rand(10, 10).astype('float32'))
y = theano.shared(np.random.rand(10, 10).astype('float32'))

ops_scalar = [round, abs, neg, sign, inv, sqrt, square, exp, log, 
               ceil, floor, sin, cos, diag, diagv, trace, determinant, matinv,
              cholesky, fft, ifft, sum, prod, max, min, argmax, argmin, mean, std, unique, where]
ops_binary = [add, sub, mul, div, pow, elmw_max, elmw_min, matmul, batch_matmul, pad, ]

ops_bool = [eq, lt, le, gt, ge, logic_and, logic_or, logic_not, logic_xor]

rand_func = [randn, rand, binomial, shuffle]

activations = [relu, softplus, softmax, tanh, sigmoid, thresholding, clip, linear]

conv = [conv2d, conv3d, max_pool2d, max_pool3d, avg_pool2d, avg_pool3d, window_slides]

loss = [mse, mae, msle, sqr_hinge, hinge, categorical_crossentropy, binary_crossentropy, cosine_proximity]

optimizer = [gd, momentum, rmsprop, adagrad, adadelta, adam, adamax]

funcs = [ops_scalar, ops_binary, ops_bool, rand_func, activations, conv, loss]

for f in ops_scalar:
  print(type(f(x)))