# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-13 10:33:42
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-20 20:36:43

import tunas.core.arch as arch
import tunas.util.basic as ubasic
import dag.DirectedAcyclicGraph as DAG

class Variable(object):
    '''
        Variable object, contains methods of get_value and set_value
    '''
    def __init__(self, name = None, is_placeholder = False,
            value = None, dtype = arch.FLOATX, shape = None):
        self.is_placeholder = is_placeholder
        if is_placeholder == False:
            self._var_ = arch.variable(value, dtype, name)
        else:
            self._var_ =  arch.placeholder(shape, None, dtype = dtype, name)

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

class Model(object):
    '''
        Model interface, micro-unit of entire model representation. Layer, and
        Sequence will subclasses. Variable is stored in a dict which contains
        key, value pairs. Method forward should be implemented to compute the
        output, while custom backward should be implemented in backward
        function, with new gradient applied to optimizer. New operators are
        implemented using C++ to define a new Operation if needed, in case
        your model is difficult for arches to figure out its gradient.
    '''
    def __init__(self, pre = None, post = None, input_shape = None, initializer = None, name = None):
        assert name != None and name != ''
        self.pre = _list(pre)
        self.post = _list(post)
        self.variables = {}
        # a mathematical expression.
        if pre == None:
            self.input_shape = [model.output_shape for model in self.pre]
        else:
            self.input_shape = input_shape
        self.key = name
        self.output = self.forward(pre)

    def _list(x):
        if x == None:
            l = None
        elif isinstance(x, list):
            l = x
        else:
            l = [x]
        return l

    @property
    def output_shape(self):
        return self.get_output_shape_for(self.input_shape)

    def forward(self, input):
        '''
            'forward' function uses self.variables as inputs, no other
            variable could be used. The output could be a variable or a list
            of variable. This method must be implemented for any model based
            subclasses.
        '''
        pass

    def backward(self, cost):
        # return the gradient of newly designed backward algorithm.
        return None


    def grad(self):
        return arch.grad([self.variables[key] for key in self.variables], output)

    def set_local_para(self, paras):
        for key, value in paras.iteritems():
            self.variables[key].set_value(value)

    def get_local_para(self):
        return {key : var.get_value() for (key, var) in self.variables.iteritems()}

    def build(self, paras, initializer):
        '''
            The build function of model, including define forward function,
            initializing parameters and connection.
        '''
        pass

    def get_output_shape_for(self, input_shape):
        # calculate output shape given input_shape which is a dict with key of
        # input layer key and value the shape of the layer. Need to implement
        # for new models.
        pass

    def get_local_shape(self):
        return {key : value.get_shape() for key, value in self.variables.iteritems()}

class Framework(DAG):
    '''
        models: list of models, with order of input, layer1, ..., layern. Output is computed using the forward of the last layer.
        Parameters:
            1. input, input layer;
            2. models, other models except input;
            3. optimizer, object of class Optimizer. Modify gradient if need before assigned here.
            4. verbose type. 0 for no log, 1 - 3 for different output level.
    '''
    def __init__(self, input, models, optimizer, lr, verbose = 0):
        self.models = {}
        for model in models:
            self.models[model.key] = model
        self.optimizer = optimizer
        self.lr = lr
        self.sess = arch.get_session()
        self.verbose = verbose
        self.sess.run(arch.init_variable([ubasic.dict2list(model.get_local_para())
            for model in self.models]))

    # @TODO: make framework more flexible
    def add(model, pre = None, post = None):
        self.models.append(model)
        if pre != None:
            pre.post.append(model)
        if post != None:
            # @TODO implement validation if model is valid.
            post.pre.append(model)

    # return mathematic expression.
    @property
    def output(self):
        return [model.output for model in models if model.post == None][0]

    @property
    def predict(input):
        # input is a dict structure.
        return arch.eval(output, paras = input)

    # input_shape is 
    def get_output_shape_for(self, input_shape):
        keys = self.topology()
        output = {}
        input_keys = [s for s in keys if s.startswith('input')]
        node_keys = [s for s in keys if not s.startswith('input')]
        for key in input_keys:
            output[key] = self.models[key].get_output_shape_for(input_shape[key])

        for key in node_keys:
            cur = self.models[key]
            output[key] = cur.get_output_shape_for(
                {p.key : output[p.key] for p in cur.pre})

        return [output[key] for key in self.models.keys() if self.models[key].post == None][0]

    def get_output_shape(self):
        return [model.output_shape for model in self.models]

    def train(self, dataset, learning_rate = 0.01, epochs = 1000,
        eps = arch.EPS, early_stopping = False, decay = 0.1, **argv):
        # dataset is an object of Trunk, with interface of next_batch.
        opt = self.optimizer.minimize(self.output)
        avg = 0
        best_before = []
        for epoch in range(epochs):
            for i in range(dataset.total_batch):
                batch = dataset.next_batch()
                if self.lr == None
                    batch[self.lr] = linear_rate
                arch.eval(opt, feed_dict = batch)
                if self.verbose > 0:
                    avg += arch.eval(self.output, feed_dict = batch) / dataset.total_batch

            best_before.append(avg)
            if argv['method'] == 1 and epoch % 50 == 0:
                # decay each depc times
                learning_rate *= decay
            elif (argv['method'] == 2 and epoch % 10 == 0 and
                        min(best_before[-10 : -1]) < best_before[-1]):
                # decay if not decrease for 5 epoch
                learning_rate *= decay
            print('Epoch %05d: cost %.9f' % (epoch, avg))

