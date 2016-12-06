# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-06-07 17:03:03
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-12-06 21:41:33

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import scipy as sp
from collections import OrderedDict
import itertools

import tunas
import tunas.arch.tensorflow_arch.tensor as tfT
import tunas.arch.tensorflow_arch.ops as tfOps
import tunas.arch.theano_arch.tensor as TT
import tunas.arch.theano_arch.ops as TOps
from tunas.util.tools import Logger, LogMode

logger = Logger()

def test_expr(A, Ops):
  a = np.random.rand(10, 10).astype(tunas.arch.env.FLOATX)
  b = np.random.rand(10, 10).astype(tunas.arch.env.FLOATX)
  var = A.variable(a, name = 'a')
  err = np.linalg.norm(A.get_value(var) - a) / a.size
  logger.log(LogMode.info, ''.join(['Task test variable alloc with ' + A.arch_name(), ' of shape ', str(A.eval(A.shape(var))), ', dim ', str(A.dims(var)), ' with error ', str(err), '...', 'passed.' if abs(err) < 1e-7 else 'not passed!']))

  A.set_value(var, b)
  err =  np.linalg.norm(A.get_value(var) - b) / b.size
  logger.log(LogMode.info, ''.join(['Task test variable setting with ' + A.arch_name(), ' with error ', str(err), '...', 'passed.' if abs(err) < 1e-7 else 'not passed!']))

  input = A.placeholder(shape = [10, 10], name = 'in')
  var = Ops.sqrt(input)
  func = A.function([input], [var])
  err = np.linalg.norm(func([a]) - np.sqrt(a)) / a.size
  logger.log(LogMode.info, ''.join([A.arch_name(), ' test function error ', str(err),
     '...', 'passed.' if err < 1e-7 else 'not passed!']))

def test_tensor_ops(A, T, AR):
  np.random.seed(2012310818)
  logger.log(LogMode.info, 'test for architecture %s' % AR)
  M = np.random.rand(10, 10).astype('float32')
  mat = T.variable(M.dot(M.T))
  mat_ = T.variable(M)
  
  v = T.variable(np.random.rand(10).astype('float32'))

  x = T.variable(np.random.rand(4, 10, 10).astype('float32'))
  y = T.variable(np.random.rand(4, 10, 10).astype('float32'))
  ops_scalarf = {'round' : A.round, 'abs' : A.abs, 'neg' : A.neg, 'sign' : A.sign,
                 'inv' : A.inv, 'sqrt' : A.sqrt,'square' : A.square, 'exp' : A.exp,
                 'log' : A.log, 'ceil' : A.ceil, 'floor' : A.floor,'sin' : A.sin,
                 'cos' : A.cos, 'tan' : A.tan, 'asin' : A.asin, 'acos' : A.acos,
                 'atan' : A.atan, 'diagv' : A.diagv, 'trace' : A.trace,
                 'determinant' : A.determinant, 'matinv' : A.matinv,
                 'sum' : A.sum, 'prod' : A.prod, 'max' : A.max, 'min' : A.min,
                 'argmax' : A.argmax, 'argmin' : A.argmin,
                 'mean' : A.mean, 'std' : A.std}
  ops_scalari = {'logit_not' : A.logic_not}
  ops_binmat = {'matmul' : A.matmul}
  ops_binf = {'add' : A.add, 'sub' : A.sub, 'mul' : A.mul, 'div' : A.div, 'pow' : A.pow,
              'elemw_max' : A.elmw_max, 'elemw_min' : A.elmw_min, 'batch_matmul' : A.batch_matmul,
              'thresholding' : A.thresholding, 'eq' : A.eq, 'neq' : A.neq, 'lt' : A.lt,
              'le' : A.le, 'gt' : A.gt, 'ge' : A.ge}
  ops_bini = {'logic_and' : A.logic_and, 'logic_or' : A.logic_or, 'logic_xor' : A.logic_xor}
  activations = {'relu' : A.relu, 'softplus' : A.softplus, 'softmax' : A.softmax,
                 'tanh' : A.tanh, 'sigmoid' : A.sigmoid, 'linear' : A.linear}
  ret = OrderedDict()
  Z = np.maximum(M, -1e3)
  Z = np.minimum(M, 1e3)
  numerator = np.exp(M)
  ret['npsoftmax'] = numerator / np.sum(numerator, axis=1).reshape((-1,1))

  print('test scalar activation')
  for key, f in activations.iteritems():
    print('test %s...' % key)
    ret[key] = T.eval(f(mat_))
    
  print('test scalar boolean')
  for key, f in ops_scalari.iteritems():
    print('test %s...' % key)
    ret[key] = T.eval(f(x > 0.5))
    
  print('test binary boolean')
  for key, f in ops_bini.iteritems():
    print('test %s...' % key)
    ret[key] = T.eval(f(y > 0.5, x < 0.5))
    
  print('test scalar float')
  for key, f in ops_scalarf.iteritems():
    print('test %s...' % key)
    ret[key] = T.eval(f(mat_))
    
  print('test binary float')
  for key, f in ops_binf.iteritems():
    print('test %s...' % key)
    ret[key] = T.eval((f(x, y)))
    
  print('test binary mat')
  for key, f in ops_binmat.iteritems():
    print('test %s...' % key)
    ret[key] = T.eval(f(mat, mat_))
  
  ret['cholesky'] = T.eval(A.cholesky(mat))
  ret['diag'] = T.eval(A.diag(v))

  val, idx, cnt = A.unique(T.variable(np.random.randint(0, 10, size = (100)).astype('int32')))
  ret['unique'] = T.eval(val)[T.eval(idx)]

  print('test clipper function')
  print('test %s...' % 'clip')
  ret['clip'] = T.eval(A.clip(T.variable(np.random.rand(10, 10).astype('float32')),
         T.variable(np.random.rand(10, 10).astype('float32') * 0.01),
         T.variable(np.random.rand(10, 10).astype('float32') * 1.1)))

  print('test BN function')
  ret['BN'] = T.eval(A.batch_normalization(mat, T.variable(0.0), 
                  T.variable(1.0),  T.variable(1.0),  T.variable(1.0)))
  return ret
  
def test_loss(T, A, AR):
  def row_dist(W):
    return np.divide(W.T, np.sum(W.T, axis = 0)).T
  np.random.seed(2012310818)
  logger.log(LogMode.info, 'test loss for arch %s' % AR)
  loss = {'mse' : A.mse, 'mae' : A.mae, 'msle' : A.msle, 'sqr_hinge' : A.sqr_hinge,
          'hinge' : A.hinge, 'cosine_proximity' : A.cosine_proximity, 'binary_crossentropy' : A.binary_crossentropy}
  ret = {}
  l = 7
  gt = T.variable(np.random.uniform((l, 2)) * 1.5)
  pred = T.variable(np.random.uniform((l, 2)) * 1.5)
  pgt = A.sigmoid(gt)
  ppred = A.sigmoid(pred)
  for key, f in loss.iteritems():
    ret[key] = T.eval(f(gt, pred))
  ret['pbinary_crossentropy'] = T.eval(A.binary_crossentropy(pgt, ppred, True))
  r = 10
  c = 5
  PR = np.random.rand(r, c)
  GT = np.random.rand(r, c)
  gt = T.variable(GT)
  pred = T.variable(PR)
  ret['categorical_crossentropy'] = T.eval(A.categorical_crossentropy(gt, pred, False))
  gt = T.variable(row_dist(GT))
  pred = T.variable(row_dist(PR))
  ret['pcategorical_crossentropy'] = T.eval(A.categorical_crossentropy(gt, pred, True))
  return ret

def to_string(lst):
  return '_'.join([str(k) for k in lst])

def test_conv_pool(T, A, AR):
  logger.log(LogMode.info, 'test convolution and pooling for architecture %s...' % AR)
  np.random.seed(2012310818)
  ret = OrderedDict()
  retk = OrderedDict()
  dims = [2, 3]
  ksizes = [3, 5]
  strides = [1, 2, 5]
  paddings = ['same', 'valid']
  dim_orders = ['tf', 'th']
  pool_sizes = [2, 3, 5]
  combs = itertools.product(dims, ksizes, strides, paddings, dim_orders)
  pools = itertools.product(dims, pool_sizes, strides, paddings, dim_orders)
  n, h, w, c, t, d = [32, 16, 16, 32, 16, 64]
  for comb in combs:
    key = to_string(comb)
    if comb[4] == 'tf':
      maps = [n] + ([t] if comb[0] == 3 else []) + [h, w, c]
      ks = ([comb[1]] if comb[0] == 3 else []) + [comb[1], comb[1], c, d]
    else:
      maps = [n, c, t, h, w] if comb[0] == 3 else [n, c, h, w]
      ks = [d, c, comb[1], comb[1], comb[1]] if comb[0] == 3 else [d, c, comb[1], comb[1]]

    fmap = np.random.rand(np.prod(maps)).reshape(*maps)
    kernel = np.random.rand(np.prod(ks)).reshape(*ks)

    if AR == 'theano':
      if comb[0] == 3:
        if comb[4] == 'tf':
          kernel = kernel[::-1, ::-1, ::-1, :, :]
        elif comb[4] == 'th':
          kernel = kernel[:, :, ::-1, ::-1, ::-1]
      else:
        if comb[4] == 'tf':
          kernel = kernel[::-1, ::-1, :, :]
        else:
          kernel = kernel[:, :, ::-1, ::-1]

    vfmap = T.variable(fmap)
    vkernel = T.variable(kernel)
    if comb[0] == 2:
      ret[key] = T.eval(A.conv2d(vfmap, vkernel, tuple([comb[2] for i in range(comb[0])]), padding = comb[3], dim_order = comb[4]))
    else:
      ret[key] = T.eval(A.conv3d(vfmap, vkernel, tuple([comb[2] for i in range(comb[0])]), padding = comb[3], dim_order = comb[4]))
    print(key, kernel.size, ret[key].shape)
    retk[key] = kernel.size

  for pool in pools:
    key = to_string(pool)
    if pool[4] == 'tf':
      maps = [n] + ([t] if pool[0] == 3 else []) + [h, w, c]
    else:
      maps = [n, c, t, h, w] if pool[0] == 3 else [n, c, h, w]

    fmap = np.random.rand(*maps)

    vfmap = T.variable(fmap)
    if pool[0] == 2:
      ret['p' + key + '_max'] = T.eval(A.max_pool2d(vfmap, (pool[1], pool[1]), (pool[2], pool[2]),
        padding = pool[3], dim_order = pool[4]))
      ret['p' + key + '_avg'] = T.eval(A.avg_pool2d(vfmap, (pool[1], pool[1]), (pool[2], pool[2]),
        padding = pool[3], dim_order = pool[4]))
    else:
      ret['p' + key + '_max'] = T.eval(A.max_pool3d(vfmap, (pool[1], pool[1], pool[1]),
        (pool[2], pool[2], pool[2]), padding = pool[3], dim_order = pool[4]))
      ret['p' + key + '_avg'] = T.eval(A.avg_pool3d(vfmap, (pool[1], pool[1], pool[1]),
        (pool[2], pool[2], pool[2]), padding = pool[3], dim_order = pool[4]))
    print('p' + key + '_max', ret['p' + key + '_max'].shape)
    print('p' + key + '_avg', ret['p' + key + '_avg'].shape)

  return ret, retk

def test_conv_pool_inner(T, A, AR):
  # n, h, w, c, t, d = [2, 15, 15, 4, 1, 4]
  n, h, w, c, t, d = [1, 7, 7, 1, 1, 1]
  meta = [n, h, w, c, t, d]
  k = 3
  stride = 3

  # test for 2d
  np.random.seed(2012310818)
  logger.log(LogMode.info, 'test inner mechanism for %s' % AR)
  # fmap = np.random.rand(*[n, h, w, c])
  # thfmap = fmap.transpose([0, 3, 1, 2])
  kernel = np.random.rand(*[k, k, c, d])
  thkernel = np.random.rand(*[d, c, k, k])

  fmap = np.arange(np.prod(meta[:-1])).reshape(n, h, w, c)
  thfmap = np.arange(np.prod(meta[:-1])).reshape(n, c, h, w)
  kernel = np.arange(np.prod([k, k, c, d])).reshape(k, k, c, d) * 0.01
  thkernel = np.arange(np.prod([k, k, c, d])).reshape(d, c, k, k) * 0.01
  print(thfmap)
  if AR == 'theano':
    kernel = kernel[::-1, ::-1, :, :]
    thkernel = thkernel[:, :, ::-1, ::-1]

  
  vmap = T.variable(fmap)
  vknl = T.variable(kernel)
  vmapth = T.variable(thfmap)
  vknlth = T.variable(thkernel)
  # convtf = T.eval(A.conv2d(vmap, vknl, (stride, stride), padding = 'same', dim_order = 'tf'))
  # convth = T.eval(A.conv2d(vmapth, vknlth, (stride, stride), padding = 'same', dim_order = 'th'))

  psize = 2

  pooltf = T.eval(A.max_pool2d(vmap, (psize, psize), (stride, stride), padding = 'same', dim_order = 'tf'))
  poolth = T.eval(A.max_pool2d(vmapth, (psize, psize), (stride, stride), padding = 'same', dim_order = 'th'))
  return [pooltf, poolth]
  


def test_backend():
  test_expr(tfT, tfOps)
  test_expr(TT, TOps)
  TFS= test_tensor_ops(tfOps, tfT, 'tensorflow')
  THS = test_tensor_ops(TOps, TT, 'theano')
  for item in TFS:
    err = np.linalg.norm(TFS[item].flatten() - THS[item].flatten())
    print('%s: %f...%s' % (item, err, 'passed.' if err < 1e-5 else 'not passed!'))

  TFL = test_loss(tfT, tfOps, 'tensorflow')
  THL = test_loss(TT, TOps, 'theano')
  for key in TFL:
    err = np.linalg.norm(TFL[key] - THL[key])
    print('test %s: %f...%s' % (key, err, 'passed.' if err < 1e-5 else 'not passed!'))

  TFC, ksizes = test_conv_pool(tfT, tfOps, 'tensorflow')
  THC, ksizes = test_conv_pool(TT, TOps, 'theano')
  for key in TFC:
    if key[0] == 'p':
      print('test %s: ' % key, [TFC[key].shape[i] - THC[key].shape[i] for i in range(len(THC[key].shape))])
    else:
      err = np.linalg.norm(TFC[key] - THC[key]) / (TFC[key].size * ksizes[key])
      print('test %s: %f...%s' % (key, err, 'passed.' if err < 1e-5 else 'not passed!'))

  # tf = test_conv_pool_inner(tfT, tfOps, 'tensorflow')
  # print(tf[0])
  # print(tf[1])

  # th = test_conv_pool_inner(TT, TOps, 'theano')
  # print(th[0])
  # print(th[1])
  # print(np.linalg.norm(tf[0] - th[0]) / tf[0].size, np.linalg.norm(tf[1] - tf[1]) / tf[0].size)
   
