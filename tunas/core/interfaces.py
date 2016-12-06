# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-12 11:42:42
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-12-03 10:23:19
import tunas.arch as arch
from tunas.util.tools import *
from .dag import Node

class Trunk(object):
  '''
    Trunk is class for dataset container.
    * batch_method is used for function object to generate a mini-batch.
    * next_batch will be a generator like function, with each call
      returning a new batch.
    * release is called to release open files and other handle of
      resources.
    * pre- and post-process are defined if needed for pre or post process
      for each mini-batch.
  '''
  def __init__(self, batch_method, path):
    raise NotImplementedError

  def next_batch(self):
    raise NotImplementedError

  @property
  def total_batch(self):
    raise NotImplementedError

  def release(self):
    raise NotImplementedError

  def preprocess(self):
    raise NotImplementedError

  def postprocess(self):
    raise NotImplementedError


def Initializer(object):
  '''
    Initializer defines interfaces for different kind of initializer. It is used as following.

      init = Initializer()
      model = Layer()
      init(model)

    The __call__ and __init__ method must be implemented for different subclass.
  '''
  def __init__(self):
    raise NotImplementedError

  def __call__(self, model):
    raise NotImplementedError
