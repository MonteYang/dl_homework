# -*- coding: utf-8 -*-
# part1: initialization

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation,\
    backward_propagation, update_parameters, predict, load_dataset,\
    plot_decision_boundary, predict_dec


def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he"):
    """

    :param X:
    :param Y:
    :param learning_rate:
    :param num_iterations:
    :param print_cost:
    :return:
    """

    grads ={}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]

    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    print(parameters)

    for i in range(num_iterations):
        a3, cache = forward_propagation(X, parameters)

        cost = compute_loss(a3, Y)

        grads = backward_propagation(X, Y, cache)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print("cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
    print(parameters)
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iteration")
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


def initialize_parameters_zeros(layer_dims):
    """

    :param layer_dims:
    :return:
    """
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def initialize_parameters_random(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 10
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

#
def initialize_parameters_he(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * \
            np.sqrt(2./ layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    # parameters = initialize_parameters_zeros([3,2,1])
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))

    train_X, train_Y, test_X, test_Y = load_dataset()

    parameters = model(train_X, train_Y, initialization="he")
    print("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)
