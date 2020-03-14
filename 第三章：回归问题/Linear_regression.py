'''
y_true = 1.477 * x + 0.089 + a
y_pre = w * x + b
loss = sum((y_pre - y_true)^2)
Minimize loss(w',b')
w' * x + b' -->  y' --> y_true
'''
'''
Linear Regression       线性回归
Logistic Regression     逻辑回归：二分类
Classification          分类
'''

import numpy as np
import random

#平均误差计算函数
def compute_error_for_line_given_points(b, w, points):
    '''
    point为二维数组，第0列为x的值，第1列为标签值
    通过(y - (w*x + b))^2计算误差，最后除以数据个数获取平均误差
    :param b: 偏差
    :param w: 权重
    :param points: 数据集
    :return: 数据集中所有数据的平均误差
    '''
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w*x + b)) ** 2
    return totalError / float(len(points))

#梯度下降求解函数
def step_gradient(b_current, w_current, points, learningRate):
    '''
    给定w,b、学习率和数据集，进行一轮梯度下降获取最新的w'和b'
    :param b_current: 目前的b
    :param w_current: 目前的w
    :param points: 数据集
    :param learningRate: 学习率
    :return: 获得新的b和w
    '''
    #初始化b和w的梯度
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    '''
    loss = ((w*x + b) - y)^2
    b_gradient = 2 * ((w*x + b) - y)
    w_gradient = 2 * x * ((w*x + b) - y)
    总共有N个数据集，将b_gradient和w_gradient求和后除N，获得数据集的平均b_gradient和w_gradient
    '''
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
        # w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
        b_gradient += (2/N) * (((w_current * x) + b_current) - y)
        w_gradient += (2/N) * x * (((w_current * x) + b_current) - y)

    #用当前的b减去b的梯度，获得新的b，w同理
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return new_b, new_w

#批量迭代更新b和w的值
def gradient_descent_runner(points, starting_b, starting_w,
                            learning_rate, num_iterations):
    '''
    传入数据集，初始b和w，学习率，迭代次数，返回更新后的b和w值
    :param points: 数据集
    :param starting_b: 初始化的b
    :param starting_w: 初始化的w
    :param learning_rate: 学习率
    :param num_iterations: 迭代次数
    :return: 更新后的b和w
    '''
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, points,learning_rate)
    return b, w
#主函数
if __name__ == '__main__':
    x = np.random.rand(100)
    # y = 1.47 * x + 0.089 + np.random.rand()
    y = 1.47 * x + 0.089
    points = np.stack((x,y),1)
    b_current = 0
    w_current = 0
    learning_rate = 0.005
    iterations = 10000
    b, w = gradient_descent_runner(points,b_current,w_current,
                                   learning_rate,iterations)
    #b:0.08952864734191575，w:1.468951165336398
    print('b:{}，w:{}'.format(b, w))




