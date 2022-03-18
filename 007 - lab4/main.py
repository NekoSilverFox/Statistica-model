# ------*------ coding: utf-8 ------*------
# @Time    : 2022/2/21 13:43
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Statistica-model
# @File    : main.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
"""
    T_0 - наработка до первого отказа       MTBF(平均无故障时间)
    P(t) - Вероятность безотказной работы   无故障运行的概率 P(t) = P(T_0 > t)
    Q(t) - Вероятность отказа               故障概率 在给定的操作条件下，在给定的操作时间内至少发生一次故障的概率 Q(t) = P(T_0 < t)
    λ(t) - Интенсивность отказов            失败率，根据统计学的定义，故障率是指在相关时间段内，故障产品的数量与正在运行的产品的平均数量的比率

"""
import numpy as np
import math
from scipy.stats import norm


def LFRS(x):
    T = 8760

    return (( (x[0] > T) & (x[1] > T) | (x[2] > T)) \
            & ((x[3] > T) & (x[4] > T)) \
            & ((x[5] > T) & (x[6] > T) | (x[7] > T) & (x[8] > T) | (x[9] > T) & (x[10] > T)))


def work(L):
    N = 9539
    m = 3
    lambdaF = [40 * pow(10, -6),
               10 * pow(10, -6),
               80 * pow(10, -6)]
    n = [3, 2, 6]
    num_broken = 0

    for k in range(0, N):
        x = []

        for i in range(0, m):
            t = []

            for j in range(0, n[i]):
                alpha = np.random.random()
                t.append(-math.log(alpha) / lambdaF[i])

            for j in range(0, L[i]):
                index_min = t.index(min(t))
                t[index_min] = t[index_min] - math.log(np.random.random()) / lambdaF[i]

            for j in range(0, n[i]):
                x.append(t[j])

        num_broken = num_broken + int(not LFRS(x))

    return 1 - num_broken / N


if __name__ == '__main__':
    # ppf = norm.ppf(q=0.999, loc=0, scale=1)  # 标准正态分布的四分位数

    p0 = 0.999
    arr_block = [0, 0, 0]
    num = 9
    for type_1 in range(1, num):
        arr_block[0] = type_1

        for type_2 in range(1, num):
            arr_block[1] = type_2

            for type_3 in range(1, num):
                arr_block[2] = type_3
                p_broken = work(arr_block)

                if p_broken > p0:
                    print('\033[32mP = ', p_broken, "  ", arr_block, "   ", sum(arr_block), '+\033[0m')
                else:
                    print("P = ", p_broken, "  ", arr_block, "   ", sum(arr_block))
