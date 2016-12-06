# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-13 10:33:42
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-12-03 21:45:40

import tunas.arch as arch
import tunas.util.tools as tools
from .dag import DirectedAcyclicGraph as DAG, EmptyKeyError, Node
from collections import deque, OrderedDict
from itertools import chain

class Variable(object):
  '''
    Variable object, contains methods of get_value and set_value
  '''
  def __init__(self, key, is_placeholder = False, value = None, 
      dtype = arch.get_floatX(), shape = None, trainable = True):
    if (key == None or key == ''):
      raise EmptyKeyError
    self.key = key
    self.is_placeholder = is_placeholder
    self.trainable = trainable
    if is_placeholder == False:
      self._var_ = arch.variable(value, dtype, key)
    else:
      self._var_ = arch.placeholder(shape, None, dtype, key)

  def get_value(self):
    return arch.get_value(self._var_)

  def set_value(self, value):
    arch.set_value(self._var_, value)

  def get_shape(self):
    return arch.shape(self._var_)

  def set_shape(self, shape):
    return arch.reshape(self._var_, shape)

  def get_dims(self):
    return len(arch.shape(self._var_))

  def cast(self, dtype):
    return tf.cast(self._var_, dtype)

class Model(Node):
  '''
    Model interface, micro-unit of entire model representation. Layer, and
    Sequence will subclasses. Variable is stored in a dict which contains
    key, value pairs. Method forward should be implemented to compute the
    output, while custom backward should be implemented in backward
    function, with new gradient applied to optimizer. New operators are
    implemented using C++ to define a new Operation if needed, in case
    your model is difficult for arches to figure out its gradient.
    memeber and data type:
      pre: list.
      post: list.
      input_shape: not None if it is input layer.
  '''
  def __init__(self, key, type, input_shape = None):
    super(Model, self).__init__(name)
    self.elements = OrderedDict()
    self.type = type
    self.initializer = initializer
    self.input_shape = input_shape

  @property
  def output_shape(self):
    return self.get_output_shape_for(self.input_shape)

  def forward(self, inputs):
    '''
      method define forward expression.
    '''
    raise NotImplementedError

  @property
  def grad(self):
    '''
      method define backward gradient, i.e., define $\frac{\partial L}{\partial f_i}$
      if grad is none, return grad from arch
      if grad is not none, return custom gradient.
    '''
    raise NotImplementedError

  def get_variables(trainable = None):
    if type(x) == Model:
      lvar = [x.get_variables(trainable) for x in self.elements.values()]
    else:
      lvar = [x for x in self.elements.values() if x.trainable == trainable]
    return lvar

  def get_output_shape_for(self, input_shape):
    # calculate output shape given input_shape which is a dict with key of
    # input layer key and value the shape of the layer. Need to implement
    # for new models.
    raise NotImplementedError

class Framework(DAG):
  '''
    models: list of models, with order of input, layer1, ..., layern. Output is computed using the forward of the last layer.
    Parameters:
      1. inputs, list of input layers;
      2. models, other models except input;
      3. optimizer, object of class Optimizer. Modify gradient if need before assigned here.
      4. verbose type. 0 for no log, 1 - 3 for different output level.
  '''
  def __init__(self, optimizer, lr, verbose = 0):
    super(Framework, self).__init__(models)
    self.models = OrderedDict()

    self.optimizer = optimizer
    self.lr = lr
    self.verbose = verbose
    self.outputs = []

  def add_model(self, x, pre = None, post = None):
    '''
      A model's pre and post can be assigned before added to the 
      framework while also able to finished during add_model. This
      enables users to modify neural networkand change network
      architectures during traing
    '''
    self.add_node(x)
    if post != None:
      for p in post:
        self.add_edge(x, p)
    if pre != None:
      for p in pre:
        self.add_edge(p, x)

  def erase_model(self, x):
    self.erase_node(x)

  def get_output(self):
    output = []
    for M in models.values():
      if len(M.post) == 0:
        output.append(M.get_output)
    return output

  def branch(x):
    if len(x.pre) == 0:
      return x.get_variables()
    else:
      incomings = [branch(p) for p in x.pre]
      return x.forward(inputs)

  # return output shape value given input shape of layer key
  def get_output_shape_for(self, key, input_shapes):
    return self.models[key].get_output_shape_for(input_shapes)

  # return output shape variable of layer key
  def get_output_shape(self, key):
    return self.models[key].get_output_shape()

  def get_variables(self, slim = True, trainable = False):
    variables = []
    models = self.models if slim == False else slim_arch()
    for k, v in models:
      variables += chain.from_iterable(v.get_variables())
    return variables

  def slim_arch(self):
    if len(self.outputs) > 0:
      q = deque()
      for v in self.outputs:
        q.push(v)
      layers = self.topology(self.outputs)
      models = [self.models[k] for k in layers if layers[k] != -1]
    else:
      models = self.models
    return models


  def get_values(slim = True, trainable = False):
    vars = self.get_variable(slim, trainable)
    vals = []
    for v in vars:
      vals.append(v.get_value())
    return vals

  def save_values(fp):
    vars = self.get_variables(slim = False)
    

  def load_values(fp):
    pass

  def load_models(fp):
    pass

  def save_models(fp):
    pass

  def draw(fp):
    pass

  def build(loss, targets, vals):
    '''
      build following items:
        1. all output function of each layer
        2. training function
        3. validation function
    '''
    pass

  def train(batchX, batchY):
    pass

  def forwards(batchX, layer):
    pass

  def predict(batchX):
    pass
