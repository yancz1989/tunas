# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-17 20:35:01
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-18 08:54:48
from __future__ import print_function, division, absolute_import
import timeit
import numpy
import tensorflow as tf
from get_data import *

# Parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
# Create model
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
    return tf.matmul(layer_2, _weights['out']) + _biases['out']

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Initializing the variables
init = tf.initialize_all_variables()

def to_one_hot(lbl, cols):
    ret = np.zeros((len(lbl), cols), dtype='float32')
    ret[:, lbl] = 1.0
    return ret

def next_batch(sample, label, idx, batch_size):
    idx = range(batch_size * i, batch_size * (i + 1))
    return sample[idx], label[idx]
# Launch the graph
dataset = load_mnist('./dataset/')
train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = [
    dataset['X_train'], dataset['y_train'], dataset['X_val'],
    dataset['y_val'], dataset['X_test'], dataset['y_test']]

train_set_y = to_one_hot(train_set_y, 10)
with tf.Session() as sess:
    sess.run(init)
    start_time = timeit.default_timer()
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train_set_x)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(train_set_x, train_set_y, i, batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    end_time = timeit.default_timer()
    print(end_time - start_time)
