# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-16 08:38:49
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-17 11:38:06


from __future__ import print_function, division, absolute_import
import timeit
import numpy

import theano
import theano.tensor as T
from get_data import *


class LogisticRegression(object):
    def __init__(self, x, n_in, n_out):
        self.W = theano.shared(value = numpy.zeros((n_in, n_out),
            dtype = theano.config.floatX), name = 'W', borrow = True)
        self.b = theano.shared(value = numpy.zeros((n_out,), dtype = theano.config.floatX),
            name='b', borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.x = x

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(path):
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    dataset = load_mnist(path)
    train_set_x, train_set_y = shared_dataset([dataset['X_train'], dataset['y_train']])
    valid_set_x, valid_set_y = shared_dataset([dataset['X_val'], dataset['y_val']])
    test_set_x, test_set_y = shared_dataset([dataset['X_test'], dataset['y_test']])

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           batch_size=600):
    datasets = load_data('./dataset/')
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('... building the model')
    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')
    classifier = LogisticRegression(x=x, n_in=28 * 28, n_out=10)
    cost = classifier.negative_log_likelihood(y)

    test_model = theano.function(inputs = [index], outputs = classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(inputs = [index], outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(inputs = [index], outputs = cost, updates = updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    print('... training the model')
    # early-stopping parameters
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)
    print(n_train_batches)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                    (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print('epoch %i, minibatch %i/%i, test error of best model %f.' % (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()
    print(end_time - start_time)


if __name__ == '__main__':
    sgd_optimization_mnist()