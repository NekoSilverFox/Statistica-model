# ------*------ coding: utf-8 ------*------
# @Time    : 2022/2/24 19:48
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Statistica-model
# @File    : main.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import math


TITLE_FONT_SIZE = 36
LABEL_FONT_SIZE = 36
TICK_FONT_SIZE = 32
FONT = 'Times New Roman'


def irnpoi(mu: int, size: int) -> np.ndarray:
    """
    【2.4.1】泊松分布（算法 1）
    :param mu: int 类型
    :param size: 数组大小
    :return: 具有泊松分布的 numpy.ndarry
    """
    if mu < 0 or size <= 0:
        raise ValueError

    arr_irnpoi = []

    for i in range(size):
        if mu < 88:
            i_uniform = np.random.uniform(low=0, high=1)
            p_t = math.exp(-mu)
            m = 1
            while (i_uniform - p_t) >= 0:
                i_uniform -= p_t
                p_t *= (mu / m)
                m += 1
            arr_irnpoi.append(m)
        else:
            m = np.random.normal(loc=mu, scale=mu, size=1)
            arr_irnpoi.append(m)

    return np.array(arr_irnpoi)


def irnpsn(mu: int, size: int) -> np.ndarray:
    """
    【2.4.1】泊松分布（算法 1）
    :param mu: int 类型
    :param size: 数组大小
    :return: 具有泊松分布的 numpy.ndarry
    """
    if mu < 0 or size <= 0:
        raise ValueError

    arr_irnpsn = []

    for i in range(size):
        if mu < 88:
            i_uniform = np.random.uniform(low=0, high=1)
            p_t = i_uniform
            m = 1
            while p_t >= math.exp(-mu):
                i_uniform = np.random.uniform(low=0, high=1)
                p_t *= i_uniform
                m += 1
            arr_irnpsn.append(m)
        else:
            m = np.random.normal(loc=mu, scale=mu, size=1)
            arr_irnpsn.append(m)

    return np.array(arr_irnpsn)


def plot_poisson_hist(arr_poisson: np.ndarray, save_path: str):
    """
    绘制直方图并保存
    :param arr_poisson: 具有泊松分布的数组
    :param save_path: 图像存储路径
    :return:
    """
    plt.figure(figsize=(18, 10), dpi=100)
    plt.hist(x=arr_poisson, bins=(arr_poisson.max() - arr_poisson.min()), density=True)
    plt.grid(linestyle='--', alpha=0.5)
    plt.title('Закон Пуассона \n'
              'λ=4 size=100',
              fontdict={'family': FONT, 'size': TITLE_FONT_SIZE})
    plt.xlabel('x', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.ylabel('P(x)', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.xticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    plt.yticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    plt.xlim(0, 15)
    plt.savefig(save_path)
    plt.show()


def chi2_independence(data: np.ndarray, alpha: float):
    """
    критерием Пирсона
    假设检验重要知识
    H0:A与B相互独立
    H1：A与B不相互独立
    若卡方值大于临界值，拒绝原假设，表示A与B不相互独立，A与B相关
    函数中re返回为1表示拒绝原假设，0表示接受原假设

    :param alpha: 置信度，用来确定临界值（уровнем значимости）
    :param data: 数据，请使用 numpy.array 数组
    :return:
        chi_2 : 卡方值，也就是统计量
        p     : P值（统计学名词），与置信度对比，也可进行假设检验，P值小于置信度，即可拒绝原假设
        dof   : 自由度
        re    : 判读变量，1表示拒绝原假设，0表示接受原假设
        expctd: 原数据数组同维度的对应理论值
    """
    chi_2, p, dof, expctd = chi2_contingency(data)

    cv = 0
    if dof == 0:
        print('自由度应该大于等于1')
    elif dof == 1:
        cv = chi2.isf(alpha * 0.5, dof)
    else:
        cv = chi2.isf(alpha * 0.5, dof-1)

    if chi_2 > cv:
        re = 1  # 表示拒绝原假设
    else:
        re = 0  # 表示接受原假设

    return chi_2, p, dof, re, expctd


if __name__ == '__main__':
    print('>' * 50, '\nirnpoi(mu=4, size=100):')
    arr_poi = irnpoi(mu=4, size=100)
    print('\tmin: ', arr_poi.min(), '\n\tmax: ', arr_poi.max(), '\n')

    chi_2, p, dof, re, expctd = chi2_independence(alpha=0.05, data=[arr_poi, arr_poi])
    print('【chi_2】卡方值：', chi_2, '\n',
          '【p】P 值：', p, '\n',
          '【dof】自由度：', dof, '\n',
          '【re】判读变量：', re, '\n',
          '【expctd】理论值：', expctd, '\n')
    plot_poisson_hist(arr_poisson=arr_poi,
                      save_path='./result/1_irnpoi.png')

    ############################################################################
    # print('\n\n', '>' * 50, '\nirnpoi(mu=4, size=100):')
    # arr_psn = irnpsn(mu=4, size=100)
    # print('\tmin: ', arr_psn.min(), '\n\tmax: ', arr_psn.max())
    #
    # chi_2, p, dof, re, expctd = chi2_independence(alpha=0.05, data=arr_psn)
    # print('【chi_2】卡方值：', chi_2, '\n',
    #       '【p】P 值：', p, '\n',
    #       '【dof】自由度：', dof, '\n',
    #       '【re】判读变量：', re, '\n',
    #       '【expctd】理论值：', expctd, '\n')
    # plot_poisson_hist(arr_poisson=arr_psn,
    #                   save_path='./result/2_irnpsn.png')

    pass
