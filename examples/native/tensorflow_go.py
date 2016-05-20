# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-16 08:39:01
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-17 09:51:49
from __future__ import print_function, division, absolute_import
import timeit
import numpy

import tensorflow as tf
from get_data import *


class LogisticRegression(object):
    def __init__(self, x, n_in, n_out):
        self.W = tf.Variable(tf.zeros(n_in, n_out))
        self.b = tf.Variable(tf.zeros(n_out, ))
        self.x = x
        self.p_y_given_x = tf.nn.softmax(tf.matmul(self.W, self.x) + b)
        self.y_pred = tf.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -tf.reduce_mean(y * tf.log(self.p_y_given_x), reduction_indices = 1)

    def errors(self, y):
        if  len(y.get_shape())!= len(self.y_pred.get_shape()):
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype in [tf.int32, tf.int16, tf.int8]:
            return tf.reduce_mean(tf.not_equal(self.y_pred, y))
        else:
            raise NotImplementedError()

def next_batch(sample, label, idx, batch_size):
    idx = batch_size * i : batch_size * (i + 1)
    return sample[idx], label[idx]

def sgd_optimization_mnist(learning_rate = 0.13, n_epochs = 1000, batch_size = 600):
    session.run(tf.initialize_all_variables())
    # load data 
    dataset = load_mnist('./dataset/')
    train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = [
        dataset['X_train'], dataset['y_train'], dataset['X_val'],
        dataset['y_val'], dataset['X_test'], dataset['y_test']]

    n_train_batches = train_set_x.shape[0] // batch_size
    n_valid_batches = valid_set_x.shape[0] // batch_size
    n_test_batches = test_set_x.shape[0] // batch_size

    n_in = 784;
    n_out = 10;
    learning_rate = 0.01
    training_epochs = 25
    batch_size = 100
    display_step = 1
    print('... building the model')

    x = tf.placeholder('float32', [None, n_in])
    y = tf.placeholder('float32', [None, n_out])
    classifier = LogisticRegression(input = x, n_in = n_in, n_out = n_out)
    cost = classifier.negative_log_likelihood(y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.initialize_all_variables()

    print('... training the model')
    # early-stopping parameters
    start_time = timeit.default_timer()

    # start training
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = next_batch(train_set_x, train_set_y, i, batch_size)
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
                avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    end_time = timeit.default_timer()
    print(end_time - start_time)

if __name__ == '__main__':
    sgd_optimization_mnist()