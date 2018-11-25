# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# np.random.seed(1)
#
# X, Y = load_planar_dataset()

# 可视化
# plt.scatter(X[0, :], X[1, :], c=Y.squeeze(), s=40, cmap=plt.cm.Spectral)
# plt.show()

# shape_X = X.shape
# shape_Y = Y.shape
# m = X.shape[1] # the number of training set

# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, Y.T)

# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
#
# plt.title("Logistic Regression")
# plt.show()
# LR_predictions = clf.predict(X.T)
# print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
#        '% ' + "(percentage of correctly labelled datapoints)")
#
# plot_decision_boundary(model=clf, X=X, y=Y)




# ---------------------------------
# Reminder: The general methodology to build a Neural Network is to:
# 1. Define the neural network structure ( # of input units, # of hidden units, etc).
# 2. Initialize the model’s parameters
# 3. Loop:
# - Implement forward propagation
# - Compute loss
# - Implement backward propagation to get the gradients
# - Update parameters (gradient descent)
#
# You often build helper functions to compute steps 1-3 and then merge them into one function we call nn_model(). Once you’ve built nn_model() and learnt the right parameters, you can make predictions on new data.
# ---------------------
# 作者：Koala_Tree
# 来源：CSDN
# 原文：https://blog.csdn.net/koala_tree/article/details/78067464
# 版权声明：本文为博主原创文章，转载请附上博文链接！

def layer_sizes(X, Y):
    '''
    定义网络的结构
    :param X: input dataset (input size, number of examples)
    :param Y: labels of shape (output size, number of examples)

    :return:
    n_x -- the size of input layer
    n_h -- the size of hidden layer
    n_y -- the size of output layer
    '''
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)

# X_assess, Y_assess =  layer_sizes_test_case()
# (n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
# print(n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    '''
    参数初始化，根据每层的单元数确定参数W、b的shape，并初始化W、b，
    注意W需要初始化成随机矩阵，不能全零

    :param n_x: size of input layer
    :param n_h: size of hidden layer
    :param n_y: size of output layer

    :return:
    params -- python dictionary containing parameters:
        W1 -- weight matrix of shape (n_h, n_x)
        b1 -- bias vector of shape (n_h, 1)
        W2 -- weight matrix of shape (n_y, n_h)
        b2 -- bias vector of shape (n_y, 1)
    '''
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    assert W1.shape == (n_h, n_x)
    assert b1.shape == (n_h, 1)
    assert W2.shape == (n_y, n_h)
    assert b2.shape == (n_y, 1)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

# n_x, n_h, n_y = initialize_parameters_test_case()
# parameters  = initialize_parameters(n_x, n_h, n_y)
# print(parameters)

def forward_propagation(X, parameters):
    """
    前向传播
    :param X:
    :param parameters:
    :return:
    A2 --
    cache --
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1 # (n_h, n_x) . (n_x, m) => (n_h, m)
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert A2.shape == (1, X.shape[1])

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)
# print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))

# logprobs = np.multiply(np.log(A2), Y)
# cost = - np.sum(logprobs)

def compute_cost(A2, Y, parameters):
    '''
    计算代价函数
    :param A2: (n_y, m), m is the number of examples
    :param Y: (1, m)
    :param parameters: python dictionary containing your parameters W1, b1, W2 and b2
    :return:
    cost -- cross-entropy cost given equation
    '''
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), (1-Y))
    cost = -(1.0/m) * np.sum(logprobs)

    cost = np.squeeze(cost)

    assert isinstance(cost, float)

    return cost

# A2, Y_assess, parameters = compute_cost_test_case()
# print("cost = " + str(compute_cost(A2, Y_assess, parameters)))

def backward_propagation(parameters, cache, X, Y):
    """

    :param parameters:
        parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    :param cache:
        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}
    :param X: shape of (n_x, m)
    :param Y: shape of (1, m)
    :return:
    """
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y # (n_y, m)
    dW2 = (1.0 / m) * np.dot(dZ2, A1.T) # (n_y, m) . (m, n_h) => (n_y, n_h)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True) # (n_y, m) => (n_y, 1)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2)) # (n_y, n_h).T . (n_y, m) * (n_h, m) => (n_h, m)
    dW1 = (1.0 / m) * np.dot(dZ1, X.T) #
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
#
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))

def update_parameters(parameters, grads, learning_rate=1.2):
    '''

    :param parameters:
        parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    :param grads:
            grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    :param learning_rate: 超参数，可调
    :return:
        parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    '''
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def nn_model(X, Y, n_h, num_iteratinos=10000, print_cost=False):
    """

    :param X:
    :param Y:
    :param n_h:
    :param numiteratinos:
    :param print_cost:
    :return:
    """
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iteratinos):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters,grads, learning_rate=1.2)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters

# X_assess, Y_assess = nn_model_test_case()
# parameters = nn_model(X_assess, Y_assess, 4, num_iteratinos=10000, print_cost=True)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions

# parameters, X_assess = predict_test_case()
#
# predictions = predict(parameters, X_assess)
# print("predictions mean = " + str(np.mean(predictions)))

if __name__ == '__main__':
    np.random.seed(1)

    X, Y = load_planar_dataset()
    parameters = nn_model(X, Y, n_h=4, num_iteratinos=10000, print_cost=True)

    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title('Decision Boundary for hidden layer size' + str(4))