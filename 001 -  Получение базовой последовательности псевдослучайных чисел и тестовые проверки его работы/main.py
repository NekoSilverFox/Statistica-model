# ------*------ coding: utf-8 ------*------
# @Time    : 2022/2/13 20:29
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Statistica-model
# @File    : main.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------

import numpy as np
from matplotlib import pyplot as plt


def math_indicators(ndarry):
    """ 计算并返回 ndarry
    - M（期望值 | математическое ожидание по результатам наблюдений）
    - D（方差 | эмпирическая дисперсия）
    - S（偏差 | отклонение）
    
    :param ndarry: 需要计算的数组
    :return: [M, M_S, D, D_S]
    """
    if ndarry is None:
        print('ndarry can not be None')
        return None

    M = ndarry.mean()
    D = ndarry.var()

    return np.array([M, 0.5 - M, D, 0.08333 - D])


def autocorrelation(ndarry):
    """
    计算自相关系数，并返回对应数组
    :param ndarry:要计算的数组
    :return: 自相关系数的 ndarry
    """
    if ndarry is None or ndarry.size == 0:
        print('Array can not be empty')
        return None

    denominator = ndarry.var() * ndarry.size  # 分母
    M = ndarry.mean()  # 方差
    size = ndarry.size
    arr_auto = []
    for f in range(size):
        numerator = sum((ndarry[:size - f] - M) * (ndarry[f:] - M)).cumsum()  # 分子
        arr_auto.append(numerator / denominator)

    print(np.array(arr_auto))
    return np.array(arr_auto)


def plot_autocorrelation(ndarry):
    """
    绘制 autocorrelation ndarry 的为柱状图
    :param ndarry: 数组
    :return: None
    """
    if ndarry is None or ndarry.size == 0:
        print('Array can not be empty')
        return None

    x = range(ndarry.size - 1)
    plt.figure(figsize=(40, 16), dpi=160)
    plt.bar(x, [1, 2, 3, 5, 6, 9, 8, 8, 1])

    plt.title('Autocorrelation')
    plt.xlabel('f')
    plt.ylabel('K(f)')
    y_ticks = np.linspace(start=-1.0, stop=1.0, num=20)
    plt.yticks(y_ticks)
    plt.grid(linestyle='--', alpha=0.5)

    plt.show()


if __name__ == '__main__':
    for n in [10, 100]:
        random_array = np.random.uniform(low=0, high=1, size=n)
        result = math_indicators(random_array)

        print('-' * 50)
        print('n = ' + n.__str__() + '\n'
              + 'M   = ' + result[0].__str__() + '\n'
              + 'M_S = ' + result[1].__str__() + '\n'
              + 'D   = ' + result[2].__str__() + '\n'
              + 'D_S = ' + result[3].__str__() + '\n\n')

        arr_auto = autocorrelation(random_array)
        plot_autocorrelation(arr_auto)

    pass
