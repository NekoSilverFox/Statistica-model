# ------*------ coding: utf-8 ------*------
# @Time    : 2022/2/24 19:48
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Statistica-model
# @File    : main.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy import stats
import st_method
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
    【2.4.2】泊松分布（算法 2）
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


def get_hist_arr(arr: np.ndarray, arr_cut_value: np.ndarray):
    """
    根据传入的数值数组进行分隔并统计分布数量
    :param arr: 数组
    :param arr_cut_value: 要分隔的区间数组；比如：[0, 3, 5, 8] 则统计：[0, 3), [3, 5), [5, 8)
    :return: 统计结束后的数组
    """
    if arr is None or arr_cut_value is None:
        raise ValueError

    return pd.cut(x=arr, bins=arr_cut_value).value_counts().values






def plot_poisson_hist(arr_poisson: np.ndarray, save_path: str):
    """
    绘制直方图并保存
    :param arr_poisson: 具有泊松分布的数组
    :param save_path: 图像存储路径
    :return:
    """
    plt.figure(figsize=(18, 10), dpi=100)
    plt.hist(x=arr_poisson, bins=10, density=True)
    plt.grid(linestyle='--', alpha=0.5)
    plt.title('Закон Пуассона \n'
              'λ=4 Size=100 Section=10',
              fontdict={'family': FONT, 'size': TITLE_FONT_SIZE})
    plt.xlabel('x', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.ylabel('P(x)', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.xticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    plt.yticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    plt.xlim(0, 15)
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':

    print('numpy 泊松分布：')
    arr_obs_poisson = np.random.poisson(lam=4, size=100)
    pd.Series(arr_obs_poisson).sort_values().plot(kind='hist')

    arr_obs_hist = get_hist_arr(arr=arr_obs_poisson, arr_cut_value=np.array(range(16)))  # [0, 16)

    print('-' * 50)














    #
    # print('>' * 50, '\nirnpoi(mu=4, size=100):')
    #
    # crit = stats.chi2.ppf(q=0.05,
    #                       df=9)
    # print('临界值：', crit)
    #
    # arr_poi = irnpoi(mu=4, size=100)
    # print('\tmin: ', arr_poi.min(), '\n\tmax: ', arr_poi.max())
    #
    # arr_pdf = st_method.get_pdf(arr_poi, cut_num=10)
    # # chi_2 = chisquare(f_obs=arr_poi)
    # chi_2 = chisquare(f_obs=arr_pdf)
    # print('chi_2: ', chi_2)
    #
    # plot_poisson_hist(arr_poisson=arr_poi,
    #                   save_path='./result/1_irnpoi.png')

    ############################################################################
    # print('\n\n', '>' * 50, '\nirnpoi(mu=4, size=100):')
    # arr_psn = irnpsn(mu=4, size=100)
    # print('\tmin: ', arr_psn.min(), '\n\tmax: ', arr_psn.max())
    #
    # chi_2 = chisquare(arr_psn)
    # print('chi_2: ', chi_2[0])
    #
    # plot_poisson_hist(arr_poisson=arr_psn,
    #                   save_path='./result/2_irnpsn.png')

    pass
