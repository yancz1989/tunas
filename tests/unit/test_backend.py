# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-06-07 17:03:03
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-06-22 17:13:47

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import scipy as sp

import tunas
import tunas.core.arch.tensorflow_arch.tensor as tfT
import tunas.core.arch.tensorflow_arch.ops as tfOps
import tunas.core.arch.theano_arch.tensor as TT
import tunas.core.arch.theano_arch.ops as TOps
from tunas.util.io.log import log

def test_expr(A):
	a = np.random.rand(10, 10).astype(tunas.core.env.FLOATX)
	b = np.random.rand(10, 10).astype(tunas.core.env.FLOATX)
	var = A.variable(a, name = 'a')
	err = np.linalg.norm(A.get_value(var) - a) / a.size
	log(['Task test variable alloc with ' + A.arch_name(), ' of shape ', A.eval(A.shape(var)),
		', dim ', A.dims(var),
		' with error ', err, '...', 'pass!' if abs(err) < 1e-7 else 'not pass!'])

	A.set_value(var, b)
	err =  np.linalg.norm(A.get_value(var) - b) / b.size
	log(['Task test variable setting with ' + A.arch_name(), ' with error ', err, '...', 'pass!' if abs(err) < 1e-7 else 'not pass!'])

	input = A.placeholder(shape = [10, 10], name = 'in')
	var = A.sqrt(input)
	func = A.function([input], [var])
	err = np.linalg.norm(func([a]) - np.sqrt(a)) / a.size
	log([A.arch_name(), ' test function error ', str(err),
		 '...', 'pass!' if err < 1e-7 else 'not pass!'])

def test_tensor_ops(A, T, AR):
  np.random.seed(2012310818)
  print('TEST for architecture %s' % AR)
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
  ret = {}
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
  return ret
  
  ret['cholesky'] = T.eval(A.cholesky(mat))
  ret['diag'] = T.eval(A.diag(v))

  val, idx, cnt = A.unique(T.variable(np.random.randint(0, 10, size = (100)).astype('int32')))
  ret['unique'] = T.eval(val)[T.eval(idx)]

  print('test clipper function')
  print('test %s...' % 'clip')
  ret[key] = T.eval(A.clip(T.variable(np.random.rand(10, 10).astype('float32')),
         T.variable(np.random.rand(10, 10).astype('float32')),
         T.variable(np.random.rand(10, 10).astype('float32'))))

  print('test BN function')
  print(type(A.batch_normalization(mat, T.variable(1.0), 
                                   T.variable(0.0),  T.variable(0.0),  T.variable(1.0))))
  
def test_loss(T, A, AR):
  def row_dist(W):
    return np.divide(W.T, np.sum(W.T, axis = 0)).T
  np.random.seed(2012310818)
  print('test loss for arch %s' % AR)
  loss = {'mse' : A.mse, 'mae' : A.mae, 'msle' : A.msle, 'sqr_hinge' : A.sqr_hinge,
          'hinge' : A.hinge, 'cosine_proximity' : A.cosine_proximity, 'binary_crossentropy' : binary_crossentropy}
  ret = {}
  l = 7
  gt = T.variable(np.random.binomial(1, 0.5, size = l))
  pred = T.variable(np.random.binomial(1, 0.5, size = l))
  for key, f in loss.iteritems():
    ret[key] = T.eval(f(gt, pred))
  ppred = A.to_one_hot(pred, 2)
  pgt = A.to_one_hot(gt, 2)
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
  ret['pcategorical_crossentropy'] = T.eval(A.categorical_crossentropy(pgt, ppred, True))
  return ret

def test_control_flow(A):
	return

def test_mlp(A):
	return

def test_conv_pool(T, A):
	return 

def test_backend():
	test_expr(tfT)
	test_expr(TT)
	TFS= test_math(tfOps, tfT, 'tensorflow')
	THS = test_math(TOps, TT, 'theano')
	for item in TFS:
  	err = np.linalg.norm(TFS[item].flatten() - THS[item].flatten()) / TFS[item].size
  	print('%s: %f.' % (item, err))

	TFL = test_loss(tfT, tfOps, 'tensorflow')
	THL = test_loss(TT, TOps, 'theano')

	for key in TFL:
	  print('test %s: %f' % (key, np.linalg.norm(TFL[key] - THL[key])))

	 
