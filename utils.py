# —*-coding:utf-8 一*一
"""
作者:LegendLeo
日期:2024年12月31日
一些工具函数
"""
import numpy as np
import matplotlib.pyplot as plt

# 加载地图
def load_map(filename):
    with open(filename, 'r') as file:
        matrix = []
        for line in file:
            # 移除字符串两端的空白字符，然后按逗号分割字符串，转换成整数后添加到matrix列表中
            row = [int(num) for num in line.strip().split(',')]
            matrix.append(row)
    return matrix

# 移动平均值
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# 绘制图像
def draw_image(return_list):
    epoch_list = list(range(len(return_list)))
    plt.plot(epoch_list, return_list)
    plt.xlabel('Epochs')
    plt.ylabel('Returns')
    plt.title('DQN for maze')
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(epoch_list, mv_return)
    plt.xlabel('Epochs')
    plt.ylabel('Returns')
    plt.title('DQN for maze (move average)')
    plt.show()

