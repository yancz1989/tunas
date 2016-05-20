# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-12 11:42:42
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-20 17:46:30

class Optimizer(object):
    '''
        Optimizer, using existing class of tensorflow. For theano,
        implementation lie in file theano/optimization. With in these
        interfaces.
            * __init__ init parameters;
            * minimize return function of update function in each step;
            * compute_gradients return gradient algebra expression of model
               w.r.t. the parameters in paras;
            * apply_gradients is used when optimization process need more
               process on gradients, like clipping or scaling. One can apply
               such operation on computed gradient and use this method to set
               the gradient in this optimizer.
    '''
    def __init__(self):
        raise NotImplementedError

    def minimize(self, func):
        raise NotImplementedError

    def compute_gradients(self, func, paras):
        raise NotImplementedError

    def apply_gradients(self, grad):
        raise NotImplementedError


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
