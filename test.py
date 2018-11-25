# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time


# def sigmoid(x):
#     """
#     compute the sigmoid of x
#     :param x: a scalar or numpy array of any size
#     :return: sigmoid(x)
#     """
#     return 1.0/(1 + np.exp(-x))
#
# def sigmoid_derivative(x):
#     """
#     求sigmoid(x)函数在x为确定值时的导数
#     :param x:
#     :return:
#     """
#     s = sigmoid(x)
#     ds = s * (1 - s)
#
#     return ds
#
# def image2vector(image):
#     """
#
#     :param image: a numpy array of shape (length, height, depth)
#
#     :return v: a vector of shape (length*height*depth, 1)
#     """
#     v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))
#
#     return v
#
# def normalizeRows(x):
#     '''
#
#     :param x:
#     :return:
#     '''
#     x_norm = np.linalg.norm(x, axis=1, keepdims=True)
#     x = x / x_norm
#
#     return x
#
# def softmax(x):
#     x_exp = np.exp(x)
#     x_sum = np.sum(x_exp, axis=1, keepdims=True) # (n, 1)
#     s = x_exp / x_sum
#
#     return s
#
# def L1(yhat, y):
#     loss = np.sum(np.abs(y - yhat))
#     return loss
class Tool(object):

    count = 0

    @classmethod
    def show_tool_count(cls):
        print(cls.count)

    def __init__(self, name):
        self.name = name

        Tool.count += 1



if __name__ == '__main__':
    tool1 = Tool("sha")
    Tool.show_tool_count()
    # a = np.arange(9).reshape((3,3))
    # print(a)
    # print(f(a))
    # print('*'*100)
    # x = np.arange(3).reshape((3,1))
    # print('x shape:', x.shape)
    # W = np.arange(12).reshape((4, 3))
    # print('W shape:', W.shape)
    # z_1 = np.dot(W, x)
    # a_1 = f(z_1)
    # print(z_1)
    # print(a_1)
    #
    # inx = np.linspace(-10,10,100)
    # y = f(inx)
    # plt.plot(inx, y)
    # plt.grid()
    # plt.show()
    # x = np.array([1,2,3])
    # print(sigmoid_derivative(x))
    # print('*'*100)
    #
    # image = np.array([[[0.67826139, 0.29380381],
    #                    [0.90714982, 0.52835647],
    #                    [0.4215251, 0.45017551]],
    #
    #                   [[0.92814219, 0.96677647],
    #                    [0.85304703, 0.52351845],
    #                    [0.19981397, 0.27417313]],
    #
    #                   [[0.60659855, 0.00533165],
    #                    [0.10820313, 0.49978937],
    #                    [0.34144279, 0.94630077]]])
    # print(image2vector(image))
    # print('*'*100)
    #
    # x = np.array([[0, 3, 4],
    #               [1, 6, 4]])
    # print(normalizeRows(x))
    # print('*'*100)
    #
    # x = np.array([
    #     [9, 2, 5, 0, 0],
    #     [7, 5, 0, 0, 0]])
    # print("softmax(x) = " + str(softmax(x)))
    # x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
    # x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
    # tic = time.process_time()
    # dot = 0
    # for i in range(len(x1)):
    #     dot += x1[i] * x2[i]
    # toc = time.process_time()
    # print('dot = ' + str(dot) + '\n --- computation time = ' + str(1000*(toc - tic)) + 'ms')
