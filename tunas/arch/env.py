# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-12 19:39:55
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-12-05 17:52:05
from __future__ import division, absolute_import, print_function

EPS = 1e-7
FLOATX = 'float32'
ARCH = 'tensorflow'
SESSION = None
DEVICE = ''

def set_device(device):
  global DEVICE
  DEVICE = device

def get_device():
  return DEVICE

def set_epsilon(eps):
  global EPS
  EPS = eps

def get_epsilon():
  return EPS
  
def set_floatX(floatX):
  global FLOATX
  FLOATX = floatX

def get_floatX():
  return FLOATX

def set_arch(arch):
  global ARCH
  ARCH = arch

def get_arch():
  return ARCH

def set_session(sess):
  global SESSION
  SESSION = sess

def get_session():
  return SESSION