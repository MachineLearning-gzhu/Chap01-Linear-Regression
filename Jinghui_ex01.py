# python
# *coding:utf-8*
# 导入包
from turtle import shape
import numpy as np  # 基本运算
import matplotlib.pyplot as plt  # 画图

# 代价函数


def hypothesis(param0, param1, data):
    """
    param0 -> θ0
    param1 -> θ1
    data   -> 训练集
    """
    m = float(len(data))
    x = data[:, 0]
    y = data[:, 1]
    # 误差平方代价函数
    cost = np.sum((param0 + param1 * x - y) ** 2) / (2 * m)
    return cost

# 批量梯度下降


def batch_gradient_descent(param0, param1, data, lr):
    """
    param0 -> θ0
    param1 -> θ1
    data   -> 训练集
    lr     -> 学习率
    """
    # 样本个数
    m = float(len(data))
    x = data[:, 0]
    y = data[:, 1]
    # 计算梯度
    dParam0 = np.sum((1/m) * (param0 + param1 * x - y))
    dParam1 = np.sum((1/m) * x * (param0 + param1 * x - y))
    # param 更新
    param0 = param0 - (lr * dParam0)
    param1 = param1 - (lr * dParam1)

    return param0, param1


def optimer(param0, param1, lr, epcoh, data):
    """
    param0 -> θ0
    param1 -> θ1
    data   -> 训练集
    lr     -> 学习率
    epcoh  -> 迭代次数
    """
    for i in range(epcoh):
        param0, param1 = batch_gradient_descent(param0, param1, data, lr)
        # 每训练100次 查看一次 代价（平方差）的情况
        if i % 100 == 0:
            print('epoch {0}:cost={1}'.format(
                i, hypothesis(param0, param1, data)))

    return param0, param1

# 绘图


def plot_data(param0, param1, data):
    """
    param0 -> θ0
    param1 -> θ1
    data   -> 训练集
    """
    x = data[:, 0]
    y = data[:, 1]
    y_prd = param0 + param1 * x

    plt.plot(x, y, 'o')
    plt.plot(x, y_prd, 'k-')
    plt.show()

# 训练主函数


def liner_regression():
    # 加载数据
    data = np.loadtxt('src/ex01/data.csv', delimiter=',')

    # 显示原始数据
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y, 'o')
    plt.show()

    # 初始化参数
    """
    param0 -> θ0
    param1 -> θ1
    data   -> 训练集
    lr     -> 学习率
    epcoh  -> 迭代次数
    """
    lr = 0.01
    epoch = 10000
    param0 = 0.0
    param1 = 0.0

    print('参数初始化如下：\n θ0 = {0}\tθ1 = {1}\n代价 cost = {2}\n'
          .format(param0, param1, hypothesis(param0, param1, data)))

    # θ0和θ1 更新
    param0, param1 = optimer(param0, param1, lr, epoch, data)

    # 输出结果
    print('最后参数为：\n 迭代次数 epoch = {0} \n θ0 = {1}\tθ1 = {2}\n代价 cost = {3}\n'
          .format(epoch, param0, param1, hypothesis(param0, param1, data)))

    # 画图
    plot_data(param0, param1, data)


if __name__ == '__main__':
    liner_regression()
