# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-05 20:59:31
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-06-21 19:59:49

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
dim3d['th'] = 'NTCHW'

kernel2d['tf'] = 'HWCD'
kernel3d['tf'] = 'THWCD'
kernel2d['th'] = 'DCHW'
kernel3d['th'] = 'DTCHW'

def _string_order(in_, out_):
  order = []
  for i in range(len(in_)):
    order.append(in_.index(out_[i]))
  return order

def _expand(dims):
  return (1, ) + dims + (1, )



