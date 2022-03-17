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
import random
import math


def LFRS(x):
    T = 8760
    # return (((x[1] > T) & (x[2] > T) | (x[3] > T)) \
    #         & ((x[4] > T) & (x[5] > T)) \
    #         & ((x[6] > T) & (x[7] > T) | (x[8] > T) & (x[9] > T) | (x[10] > T) & (x[11] > T)))
    return (((x[0] > T) & (x[1] > T) | (x[2] > T)) \
            & ((x[3] > T) & (x[4] > T)) \
            & ((x[5] > T) & (x[6] > T) | (x[7] > T) & (x[8] > T) | (x[9] > T) & (x[10] > T)))


def work(L):
    N = 3301299
    m = 3
    lambdaF = [40 * pow(10, -6),
               10 * pow(10, -6),
               80 * pow(10, -6)]
    n = [3, 2, 6]
    d = 0

    for k in range(0, N):
        x = []
        for i in range(0, m):
            t = []

            for j in range(0, n[i]):
                alpha = random.random()
                t.append(-math.log(alpha) / lambdaF[i])

            for j in range(0, L[i]):
                l = t.index(min(t))
                t[l] = t[l] - math.log(random.random()) / lambdaF[i]

            for j in range(0, n[i]):
                x.append(t[j])

        if not LFRS(x):
            d = d + 1

    return 1 - d / N


if __name__ == '__main__':
    p0 = 0.99
    L = [0, 0, 0]

    for i in range(1, 4):
        L[0] = i
        for j in range(1, 3):
            L[1] = j
            for k in range(1, 7):
                L[2] = k
                P = work(L)
                if (P > p0):
                    print('\033[32mP = ', P, "  [", L, "]   ", sum(L), '+\033[0m')
                else:
                    print("P = ", P, "  [", L, "]   ", sum(L))
