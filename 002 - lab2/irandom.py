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


def irnuni(low: float, high: int, size: int) -> np.ndarray:
    """
    标准分布
    :param low: 最小值
    :param high: 最大值
    :param size: 数组大小
    :return: 具有标准分布的 numpy.ndarry
    """
    if low < 0 or high < 0 or low > high or size <= 0:
        raise ValueError

    arr_irnuni = []

    for i in range(size):
        i_uniform = np.random.uniform(low=0, high=1, size=1)
        var = np.round((high - low + 1) * i_uniform + low)
        arr_irnuni.append(var)

    return np.array(arr_irnuni)


def irnbnl(n: int, p: float, size: int) -> np.ndarray:
    """
    二项分布
    :param n: [0, ] int
    :param p: [0, 1] float
    :param size: 数组大小
    :return: 具有二项分布的 numpy.ndarry
    """
    if n < 0 or p < 0 or p > 1 or size <= 0:
        raise ValueError

    arr_irnbnl = []

    for i in range(size):
        if n >= 100:
            m = round(np.random.normal(loc=n * p, scale=math.sqrt(n * p * (1 - p)), size=1) + 0.5)
            arr_irnbnl.append(m)
        else:
            i_uniform = np.random.uniform(low=0, high=1, size=1)
            p_r = (1 - p) ** n
            m = 0
            while (i_uniform - p_r) >= 0:
                i_uniform -= p_r
                p_r *= ((p * (n - m)) / ((m + 1) * (1 - p)))
                m += 1
            arr_irnbnl.append(m)

    return np.array(arr_irnbnl)


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

        arr_irngeo.append(j + 1)

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
        i_uniform = np.random.uniform(low=0, high=1, size=1)
        # j = round(math.log(i_uniform, math.e) / math.log(p, math.e) + 1)
        j = int(math.log(i_uniform, math.e) / math.log(p, math.e) + 1)
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
