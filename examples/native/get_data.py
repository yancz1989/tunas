# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-16 09:03:42
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-05-16 17:39:26

import os
import os.path
import gzip
import sys
import numpy as np

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_mnist(path, files):
    mkdir(path)
    for file in files:
        fpath = path + file
        if not os.path.exists(fpath):
            print('Downloading %s...' % file)
            urlretrieve('http://yann.lecun.com/exdb/mnist/' + file, fpath)

def load_mnist(path):
    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    offsets = [16, 8, 16, 8]
    get_mnist(path, files)
    data = []
    for file, offset in zip(files, offsets):
            with gzip.open(path + file, 'rb') as f:
                data.append(np.frombuffer(f.read(), np.uint8, offset = offset))
    X_train, y_train, X_test, y_test = data
    X_train = X_train.reshape(len(y_train), 784) / np.float32(256)
    X_test = X_test.reshape(len(y_test), 784) / np.float32(256)
    return {'X_train' : X_train[0 : len(X_train) - 10000],
            'X_val' : X_train[-10000 :],
            'X_test' : X_test,
            'y_train' : y_train[0 : len(X_train) - 10000],
            'y_val' : y_train[-10000 :],
            'y_test' : y_test,
            }