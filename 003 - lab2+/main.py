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
import scipy.stats
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
            arr_irnpoi.append(m - 1)
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
            arr_irnpsn.append(m - 1)
        else:
            m = np.random.normal(loc=mu, scale=mu, size=1)
            arr_irnpsn.append(m)

    return np.array(arr_irnpsn)


def get_poisson_chi_square(arr_source: np.ndarray, source_mu: int) -> int:
    """
    数组相对于标准泊松分布的卡方值
    :param source_mu: 参数泊松分布数组的 mu
    :param arr_source: 泊松分布数组
    :return: 相对于泊松分布的卡方值
    """
    if arr_source is None:
        raise ValueError

    N = arr_source.size  # 元素数量
    n = arr_source.max() - arr_source.min()  # 分的区间数
    chi_square = 0  # 卡方值
    arr_p_exp = get_arr_exp_hist_poisson(mu=source_mu, skip=n)
    arr_p_obs = get_hist_arr(arr=arr_source, arr_cut_value=n) / N

    for k in range(n):
        chi_square += (arr_p_obs[k] - arr_p_exp[k]) ** 2 / arr_p_exp[k]

    chi_square *= N

    print('原数组:', arr_source, ' (Min:', arr_source.min(), ' Max:', arr_source.max(), ')')
    print('元素数量 N:', N)
    print('区间数 n:', n)
    print('标准泊松频率数组:', arr_p_exp, ' shape:', arr_p_exp.shape, 'p_sum:', arr_p_exp.sum())
    print('随机泊松频率数组:', arr_p_obs, ' shape:', arr_p_obs.shape, 'p_sum:', arr_p_obs.sum())
    print('[EXP] Chi-square:', chi_square)
    print('[OBS] Chi-square:', scipy.stats.chi2.isf(q=0.05, df=n - 1))

    return chi_square


def get_arr_exp_hist_poisson(mu: int, skip: int) -> np.ndarray:
    """
    生成标准泊松分布频率的数组
    :param skip: 分割为几个区间
    :param mu: 参数 int 类型
    :return: ndarry 类型的数组
    """
    if mu < 0 or skip < 0:
        raise ValueError

    arr_poisson = []
    for k in range(skip):
        val = ((math.e ** -mu) * (mu ** k)) / math.factorial(k)
        arr_poisson.append(val)

    return np.array(arr_poisson)


def get_hist_arr(arr: np.ndarray, arr_cut_value: np.ndarray) -> np.ndarray:
    """
    根据传入的数值数组进行分隔并统计分布数量
    :param arr: 数组
    :param arr_cut_value: 要分隔的区间数组；比如：[0, 3, 5, 8] 则统计：(0, 3], (3, 5], (5, 8]
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
              'λ=4 Size=100 Section=' + arr_poisson.max().__str__(),
              fontdict={'family': FONT, 'size': TITLE_FONT_SIZE})
    plt.xlabel('x', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.ylabel('P(x)', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.xticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    plt.yticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    plt.xlim(0, 15)
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    print('>' * 50, '\nirnpoi(mu=4, size=100):')
    arr_poi = irnpoi(mu=4, size=100)
    chi_square = get_poisson_chi_square(arr_source=arr_poi, source_mu=4)
    plot_poisson_hist(arr_poisson=arr_poi,
                      save_path='./result/1_irnpoi.png')
    print('-' * 50)

    #############################################################################

    print('\n\n', '>' * 50, '\nirnpoi(mu=4, size=100):')
    arr_psn = irnpsn(mu=4, size=100)
    chi_square = get_poisson_chi_square(arr_source=arr_psn, source_mu=4)
    plot_poisson_hist(arr_poisson=arr_psn,
                      save_path='./result/2_irnpsn.png')
    print('-' * 50)

    pass
