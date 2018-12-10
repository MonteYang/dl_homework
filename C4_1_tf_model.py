# -*- coding: utf-8 -*-
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops

from cnn_utils import *

np.random.seed(1)

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """

    :param n_H0:
    :param n_W0:
    :param n_C0:
    :param n_y: number of classes

    :return:
    X -- placeholder for the data input, of shape (None, n_H0, n_W0, n_C0)
    Y -- placeholder for the input labels, of shape (None, n_y)
    """
    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, n_y))

    return X, Y


def initialize_parameters():
    """
    Initializes weight parameters
        W1:(4, 4, 3, 8)
        W2:(2, 2, 8, 16)
    :return:
        :parameters -- a dictionary of tensors containing W1, W2
    """
    tf.set_random_seed(1)
    W1 = tf.get_variable(name="W1", shape=(4,4,3,8),
                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable(name="W2", shape=(2,2,8,16),
                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {"W1":W1,
                  "W2":W2}

    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    Z1 = tf.nn.conv2d(X, W1, strides=)



if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # index = 6
    # plt.imshow(X_train_orig[index])
    # plt.show()

    X_train = X_train_orig / 255
    X_test = X_test_orig / 255
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    # X, Y = create_placeholders(64, 64, 3, 6)
    # print("X = " + str(X))
    # print("Y = " + str(Y))

    # tf.reset_default_graph()
    # with tf.Session() as sess_test:
    #     parameters = initialize_parameters()
    #     init = tf.global_variables_initializer()
    #     sess_test.run(init)
    #     print("W1 = " + str(parameters["W1"].eval()[1, 1, 1]))
    #     print("W2 = " + str(parameters["W2"].eval()[1, 1, 1]))

