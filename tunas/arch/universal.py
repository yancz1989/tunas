# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-05 20:59:31
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-12-06 16:29:50

# This file implement expression module for tunas, including variable,
# placeholder, function and their base support interfaces.

dim2d = {}
dim3d = {}
kernel2d = {}
kernel3d = {}

dim_orders = ['tf', 'th']
paddings = ['same', 'valid']

dim2d['tf'] = 'NHWC'
dim3d['tf'] = 'NTHWC'
dim2d['th'] = 'NCHW'
dim3d['th'] = 'NCTHW'

kernel2d['tf'] = 'HWCD'
kernel3d['tf'] = 'HWTCD'
kernel2d['th'] = 'DCHW'
kernel3d['th'] = 'DCHWT'

def _string_order(in_, out_):
  order = []
  for i in range(len(in_)):
    order.append(in_.index(out_[i]))
  return order

def _expand(dims, order):
  if order == 'th':
    return (1, 1) + dims
  else:
    return (1, ) + dims + (1,)



