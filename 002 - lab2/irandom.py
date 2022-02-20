# ------*------ coding: utf-8 ------*------
# @Time    : 2022/2/20 13:55
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Statistica-model
# @File    : irandom.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------

import numpy as np
import math


def irngeo_1(p: float, size: int) -> np.ndarray:
    """
    【2.3.1】几何分布，算法 1
    :param p: [0, 1] 的 float 类型
    :param size: 数组大小
    :return: 具有几何分布的 numpy.ndarry
    """
    if p < 0 or p > 1 or size <= 0:
        raise ValueError

    arr_irngeo = []

    for i in range(size):
        i_uniform = np.random.uniform(low=0, high=1)
        p_it = p
        j = 1

        while (i_uniform - p_it) >= 0:
            i_uniform -= p_it
            p_it *= (1 - p)
            j += 1

        arr_irngeo.append(j)

    return np.array(arr_irngeo)


def irngeo_2(p: float, size: int) -> np.ndarray:
    """
    【2.3.2】几何分布，算法 2
    :param p: [0, 1] 的 float 类型
    :param size: 数组大小
    :return: 具有几何分布的 numpy.ndarry
    """
    if p < 0 or p > 1 or size <= 0:
        raise ValueError

    arr_irngeo = []

    for i in range(size):
        i_uniform = np.random.uniform(low=0, high=1)
        j = 0

        while i_uniform > p:
            i_uniform = np.random.uniform(low=0, high=1)
            j += 1

        arr_irngeo.append(j)

    return np.array(arr_irngeo)

def irngeo_3(p: float, size: int) -> np.ndarray:
    """
    【2.3.3】几何分布，算法 3
    :param p: [0, 1] 的 float 类型
    :param size: 数组大小
    :return: 具有几何分布的 numpy.ndarry
    """
    if p < 0 or p > 1 or size <= 0:
        raise ValueError

    arr_irngeo = []

    for i in range(size):
        i_uniform = np.random.uniform(low=0, high=1)
        j = round(math.log10(i_uniform) / math.log10(1 - p)) + 1
        arr_irngeo.append(j)

    return np.array(arr_irngeo)


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
            p_t = math.exp(-10)
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
