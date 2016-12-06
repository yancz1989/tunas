# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-20 14:19:18
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-11-23 17:08:26

from collections import deque, OrderedDict

class EmptyKeyError(Exception):
  def __init__(self, message):
    super(EmptyKeyError, self).__init__('Empty variable name found. Please add one!')

class Node(object):
  def __init__(self, key):
    if key == None or key == '':
      raise EmptyKeyError
    self.inputs = []
    self.outputs = []
    self.key = key

class DirectedAcyclicGraph(object):
  '''
    Inside Node class, name is the unique global identity within group of nodes. You must make sure there are only one node with in a dag named as you give.
    Initial with a dict of nodes.
  '''
  def __init__(self, nodes):
    if nodes != None:
      self.nodes = {k : nodes[k] for k in nodes}
    else:
      self.nodes = OrderedDict()

  def add_node(self, x):
    nodes[x.key] = x

  # add edge u->v
  def add_edge(self, u, v):
    u.outputs.append(v)
    v.inputs.append(u)

  def erase_node(self, x):
    x.inputs.outputs.remove(x)
    x.outputs.inputs.remove(x)

  def erase_edge(self, u, v):
    u.outputs.remove(v)
    v.inputs.remove(u)

  def topology(output_layers = None):
    # find out whether current nodes form a dag.
    stack = deque()
    layers = OrderedDict({key : -1 for key in self.nodes.keys()})
    if output_layers == None:
      for v in output_layers:
        stack.push(v)
        layers[v.key] = 0
    else:
      for k, v in self.nodes.iteritems():
        if len(v.outputs) == 0:
          stack.push(v)
          layers[k] = 0
    while stack:
      u = stack[-1]
      stack.pop()
      for v in u.inputs:
        if visited[v.key] != -1:
          layers[v.key] = max(layers[v.key], layers[u.key] + 1)
          stack.append(v)
        else:
          raise Exception('Circle detected in node ' + v.key + '.')
    return layers