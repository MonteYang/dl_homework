# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
# from .optimization.optimize.testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def update_parameters_with_gd_test_case():
    np.random.seed(1)
    learning_rate = 0.01
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 3)
    b2 = np.random.randn(3, 1)

    dW1 = np.random.randn(2, 3)
    db1 = np.random.randn(2, 1)
    dW2 = np.random.randn(3, 3)
    db2 = np.random.randn(3, 1)

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return parameters, grads, learning_rate


def random_mini_batches_test_case():
    np.random.seed(1)
    mini_batch_size = 64
    X = np.random.randn(12288, 148)
    Y = np.random.randn(1, 148) < 0.5
    return X, Y, mini_batch_size


def initialize_velocity_test_case():
    np.random.seed(1)
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 3)
    b2 = np.random.randn(3, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


def update_parameters_with_momentum_test_case():
    np.random.seed(1)
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 3)
    b2 = np.random.randn(3, 1)

    dW1 = np.random.randn(2, 3)
    db1 = np.random.randn(2, 1)
    dW2 = np.random.randn(3, 3)
    db2 = np.random.randn(3, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    v = {'dW1': np.array([[0., 0., 0.],
                          [0., 0., 0.]]), 'dW2': np.array([[0., 0., 0.],
                                                           [0., 0., 0.],
                                                           [0., 0., 0.]]), 'db1': np.array([[0.],
                                                                                            [0.]]),
         'db2': np.array([[0.],
                          [0.],
                          [0.]])}
    return parameters, grads, v


def initialize_adam_test_case():
    np.random.seed(1)
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 3)
    b2 = np.random.randn(3, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


def update_parameters_with_adam_test_case():
    np.random.seed(1)
    v, s = ({'dW1': np.array([[0., 0., 0.], [0., 0., 0.]]),
             'dW2': np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
             'db1': np.array([[0.], [0.]]),
             'db2': np.array([[0.], [0.], [0.]])},
            {'dW1': np.array([[0., 0., 0.], [0., 0., 0.]]),
             'dW2': np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
             'db1': np.array([[0.], [0.]]),
             'db2': np.array([[0.], [0.], [0.]])})
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 3)
    b2 = np.random.randn(3, 1)

    dW1 = np.random.randn(2, 3)
    db1 = np.random.randn(2, 1)
    dW2 = np.random.randn(3, 3)
    db2 = np.random.randn(3, 1)

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return parameters, grads, v, s

def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W"+ str(l+1)] - learning_rate * grads["dW"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b"+ str(l+1)] - learning_rate * grads["db"+str(l+1)]

    return parameters

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """

    :param X: 数据集， of shape (input size, number of examples)
    :param Y: true "label" vector , of shape (1, number of examples)
    :param mini_batch_size: size of mini-batches, integer

    :return:
    mini_batches -- list , element - (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    num_complete_minibatches = math.floor(m / mini_batch_size)

    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# Momentum

def initialize_velocity(parameters):
    """
    Initialize the velocity as a python dictionary with:
        - keys: "dW1", "db1", ..., "dWL", "dbL"
        - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters

    :param parameters: python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl

    :return:
        v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl

    """
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)

    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """

    Update parameters using Momentum

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- python dictionary containing your updated velocities

    """
    L = len(parameters) // 2

    for l in range(L):
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]

    return parameters, v


# Adam
def initialize_adam(parameters):
    """

    :param parameters:
    :return:
    """
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=):
    """

    :param parameters:
    :param grads:
    :param v:
    :param s:
    :param t:
    :param learning_rate:
    :return:
    """




if __name__ == '__main__':
    # parameters, grads, learning_rate = update_parameters_with_gd_test_case()
    #
    # parameters = update_parameters_with_gd(parameters, grads, learning_rate)
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))
    # X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
    # mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)
    #
    # print("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    # print("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    # print("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    # print("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    # print("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
    # print("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    # print("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))
    # parameters = initialize_velocity_test_case()
    #
    # v = initialize_velocity(parameters)
    # print("v[\"dW1\"] = " + str(v["dW1"]))
    # print("v[\"db1\"] = " + str(v["db1"]))
    # print("v[\"dW2\"] = " + str(v["dW2"]))
    # print("v[\"db2\"] = " + str(v["db2"]))
    parameters, grads, v = update_parameters_with_momentum_test_case()

    parameters, v = update_parameters_with_momentum(parameters, grads, v, beta=0.9, learning_rate=0.01)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))
