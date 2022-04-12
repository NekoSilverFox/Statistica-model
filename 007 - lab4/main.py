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


def LFRS(x: list) -> bool:
    """
    【ЛФРС системы】根据各个部位零件的相连和正常的工作时长，判断该系统是否可以正常运行
    :param x: 包含对应零件正常运行时长的 list
    :return: bool
    """
    T = 8760  # 系统正常运行的小时数

    # ЛФРС системы
    return (( (x[0] > T) & (x[1] > T) | (x[2] > T)) \
            & ((x[3] > T) & (x[4] > T)) \
            & ((x[5] > T) & (x[6] > T) | (x[7] > T) & (x[8] > T) | (x[9] > T) & (x[10] > T)))


def start_test(arr_L: list) -> float:
    """
    开始模拟测试
    :param arr_L:
    :return:
    """
    N = 9539
    m = 3  # 系统中有几类零件
    arr_num_every_part = [3, 2, 6]  # 各个部位所包含的零件数量
    num_broken = 0
    arr_lambda = [40 * pow(10, -6),
                  10 * pow(10, -6),
                  80 * pow(10, -6)]  # 各个部位损毁的概率

    for k in range(0, N):
        x = []

        for i in range(0, m):
            t = []

            for j in range(0, arr_num_every_part[i]):
                alpha = np.random.random()
                t.append(-math.log(alpha) / arr_lambda[i])

            for j in range(0, arr_L[i]):
                index_min = t.index(min(t))
                t[index_min] = t[index_min] - math.log(np.random.random()) / arr_lambda[i]

            for j in range(0, arr_num_every_part[i]):
                x.append(t[j])

        num_broken = num_broken + int(not LFRS(x))

    return 1 - num_broken / N


if __name__ == '__main__':
    # ppf = norm.ppf(q=0.999, loc=0, scale=1)  # 标准正态分布的四分位数
    while True:
        print('>' * 100)
        str_res = '>' * 100 + '\n'

        p0 = 0.999  # 系统运行指定时长且不损坏的概率
        arr_block = [0, 0, 0]
        min_part = 3  # 每种零件的最小零件数
        max_part = 9  # 每种零件的最大零件数
        res_p = 1.0  # 存储结果中的最小值
        res_arr_part = [max_part, max_part, max_part]  # 存储结果中的零件数量

        for num_type_1 in range(min_part, max_part):
            arr_block[0] = num_type_1

            for num_type_2 in range(min_part, max_part):
                arr_block[1] = num_type_2

                for num_type_3 in range(min_part, max_part):
                    arr_block[2] = num_type_3
                    p_broken = start_test(arr_block)

                    # 调用、存储、输出
                    str = f'P = {p_broken}\t{arr_block}\t{sum(arr_block)}\n'
                    str_res += str
                    if p_broken > p0:
                        if sum(arr_block) < sum(res_arr_part):
                            res_p = p_broken
                            res_arr_part = arr_block[:]
                        # print('\033[32mP = ', p_broken, "  ", arr_block, "   ", sum(arr_block), '+\033[0m')

                    else:
                        pass
                        # print("P = ", p_broken, "  ", arr_block, "   ", sum(arr_block))

        str_res += f'\n最优结果为：\nP = {res_p}\t{res_arr_part}\t{sum(res_arr_part)}\n'
        print('\033[32m\n最优结果为：\nP = ', res_p, "  ", res_arr_part, "   ", sum(res_arr_part), '\n\033[0m')

        if sum(res_arr_part) < 15:
            # print(str_res)
            with open('out.txt', 'w+') as f:  # 设置文件对象
                f.write(str_res)  # 将字符串写入文件中
            break
