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
        # var = np.round((high - low) * i_uniform + low)
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
        i_uniform = np.random.uniform(low=0, high=1)
        j = round(math.log(i_uniform) / math.log(1 - p)) + 1
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


def irnnrm_1(size: int) -> np.ndarray:
    """
    CN：正态分布（精确转换法）
    RU：Нормальное распределение
    RNRM1  - метод Бокс-Миллера (метод точного преобразования)
    :param size: 数组大小（随机数数量）
    :return: 具有正态分布的 numpy.ndarry
    """
    if size <= 0:
        raise ValueError

    arr_irnnrm = []

    for i in range(size):
        arr_uniform = np.random.uniform(low=0, high=1, size=2)
        var = math.sqrt(-2 * math.log(arr_uniform[0])) * math.cos(2 * math.pi * arr_uniform[1])
        arr_irnnrm.append(var)

    return np.array(arr_irnnrm)


def irnnrm_2(size: int) -> np.ndarray:
    """
    CN：正态分布（基于中心极限定理的方法。）
    RU：Нормальное распределение
    RNRM2  - метод, основанный на центральной предельной теореме,
    :param size: 数组大小（随机数数量）
    :return: 具有正态分布的 numpy.ndarry
    """
    if size <= 0:
        raise ValueError

    arr_irnnrm = []

    for i in range(size):
        arr_uniform = np.random.uniform(low=0, high=1, size=14)
        var = arr_uniform.sum() - 6
        arr_irnnrm.append(var)

    return np.array(arr_irnnrm)


def irnexp(beta: int, size: int) -> np.ndarray:
    """
    CN：指数分布
    RU：Экспоненциальное распределение
    :param beta: int 类型的参数（指数）
    :param size: 数组大小（随机数数量）
    :return: 具有正态分布的 numpy.ndarry
    """
    if size <= 0:
        raise ValueError

    arr_irnexp = []

    for i in range(size):
        i_uniform = np.random.uniform(low=0, high=1, size=1)
        var = -beta * math.log(i_uniform)
        arr_irnexp.append(var)

    return np.array(arr_irnexp)


def irnchis(n: int, size: int) -> np.ndarray:
    """
    CN：智平方分布（卡方分布）
    RU：Хи-Квадрат Распределение
    :param n: int 类型的参数（指数）
    :param size: 数组大小（随机数数量）
    :return: 具有卡方分布的 numpy.ndarry
    """
    if size <= 0:
        raise ValueError

    arr_irnchis = []

    for i in range(size):
        fl_arr = np.array(0.0)
        for j in range(n):
            fl_arr = np.hstack([fl_arr, np.array(irnnrm_1(size=1)[0])])

        var = (fl_arr ** 2).sum()
        # print(var)
        arr_irnchis.append(var)

    return np.array(arr_irnchis)


def irnstud(n: int, size: int) -> np.ndarray:
    """
    CN：学生t-分布
    RU：Распределение Стьюдента
    :param n: int 类型的参数（指数）
    :param size: 数组大小（随机数数量）
    :return: 具有学生t-分布的 numpy.ndarry
    """
    if size <= 0:
        raise ValueError

    arr_irnstud = []

    for i in range(size):
        i_uniform = irnnrm_1(size=1)[0]
        i_hit = irnchis(n, size=1)[0]
        var = i_uniform / math.sqrt(i_hit / n)
        arr_irnstud.append(var)

    return np.array(arr_irnstud)