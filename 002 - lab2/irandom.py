# ------*------ coding: utf-8 ------*------
# @Time    : 2022/2/20 13:55
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Statistica-model
# @File    : irandom.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------

import numpy as np


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

