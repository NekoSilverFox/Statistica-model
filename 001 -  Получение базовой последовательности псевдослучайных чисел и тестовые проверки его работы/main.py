# ------*------ coding: utf-8 ------*------
# @Time    : 2022/2/13 20:29
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Statistica-model
# @File    : main.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------

import numpy as np
from matplotlib import pyplot as plt

TITLE_FONT_SIZE = 42
LABEL_FONT_SIZE = 36
TICK_FONT_SIZE = 22
FONT = 'Times New Roman'


def math_indicators(ndarry: np.ndarray) -> np.ndarray:
    """ 计算并返回 ndarry
    - M（期望值 | математическое ожидание по результатам наблюдений）
    - D（方差 | эмпирическая дисперсия）
    - S（偏差 | отклонение）
    
    :param ndarry: 需要计算的数组
    :return: [M, M_S, D, D_S]
    """
    if ndarry is None:
        print('ndarry can not be None')
        return None

    M = ndarry.mean()
    D = ndarry.var()

    return np.array([M, 0.5 - M, D, 0.08333 - D])


def autocorrelation(ndarry: np.ndarray) -> np.ndarray:
    """
    计算自相关系数，并返回对应数组
    :param ndarry:要计算的数组
    :return: 自相关系数的 ndarry
    """
    if ndarry is None or ndarry.size == 0:
        print('Array can not be empty')
        return None

    denominator = ndarry.var() * ndarry.size  # 分母
    M = ndarry.mean()  # 方差
    size = ndarry.size
    arr_auto = []
    for f in range(1, size):
        numerator = sum((ndarry[:size - f] - M) * (ndarry[f:] - M))  # 分子
        arr_auto.append(numerator / denominator)

    # print(np.array(arr_auto))
    return np.array(arr_auto)


def plot_autocorrelation(ndarry: np.ndarray) -> np.ndarray:
    """
    绘制 autocorrelation ndarry 的为柱状图
    :param ndarry: 数组
    :return: None
    """
    if ndarry is None or ndarry.size == 0:
        print('Array can not be empty')
        return None

    x = range(1, ndarry.size + 1, 1)
    plt.figure(figsize=(32, 16), dpi=160)
    print(len(x))
    print(ndarry)
    plt.bar(x, ndarry)

    plt.title('Autocorrelation    n=' + (ndarry.size + 1).__str__(),
              fontdict={'family': FONT, 'size': TITLE_FONT_SIZE})
    plt.xlabel('f', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.ylabel('K(f)', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.xticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    plt.yticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    # y_ticks = np.linspace(start=-1.0, stop=1.0, num=20)
    # plt.yticks(y_ticks)
    plt.grid(linestyle='--', alpha=0.5)

    plt.savefig('./result/' + (ndarry.size + 1).__str__() + '.png')
    plt.show()


def get_cdf(ndarry: np.ndarray) -> np.ndarray:
    """
    累积分布函数（英语：Cumulative Distribution Function、CDF），又叫分布函数，是概率密度函数的积分，能完整描述一个实随机变量X的概率分布。
    F(x)是分布函数，F(x)的取值表示变量小于x的概率是多少。比如说F(x)=0.85，x=0.7，那么意思是随机变量小于0.7的概率为0.85

    Ru：Эмпирическая интегральная функция распределения
    :param ndarry: [0, 1] 的随机变量值数组
    :return: 构建的分布函数 cdf ndarry
    """
    if ndarry is None or ndarry.size == 0:
        print('Array can not be empty')
        return None

    arr_step = np.linspace(start=1, stop=0, num=100)
    arr_cdf = []
    for i in arr_step:
        arr_cdf.append(np.count_nonzero(ndarry > i) / ndarry.size)

    # print(arr_cdf)
    return np.array(arr_cdf)


def plot_cdf(ndarry: np.ndarray):
    """
    绘制 cdf 为散点图
    :param ndarry:  cdf ndarry
    :return: None
    """
    if ndarry is None or ndarry.size == 0:
        print('Array can not be empty')
        return None

    x = np.linspace(start=0, stop=1, num=100)
    plt.figure(figsize=(32, 16), dpi=160)
    plt.scatter(x, ndarry)
    plt.title('Cumulative Distribution Function\n'
              'Number of sampling points=' + ndarry.size.__str__(),
              fontdict={'family': FONT, 'size': TITLE_FONT_SIZE})
    plt.xlabel('x - Значение аргумента', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.ylabel('F(x) - Вероятность', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.xticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    plt.yticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    # y_ticks = np.linspace(start=-1.0, stop=1.0, num=20)
    # plt.yticks(y_ticks)
    plt.grid(linestyle='--', alpha=0.5)

    plt.savefig('./result/cdf_' + ndarry.size.__str__() + '.png')
    plt.show()


def get_pdf(ndarry: np.ndarray, cut_num: int) -> np.ndarray:
    """ 获取概率密度函数的 ndarry
    在数学中，连续型随机变量的概率密度函数（Probability density function，简写作PDF [1]），
    在不致于混淆时可简称为密度函数，是一个描述这个随机变量的输出值，在某个确定的取值点附近的可能性的函数。
    图中，横轴为随机变量的取值，纵轴为概率密度函数的值，而随机变量的取值落在某个区域内的概率为概率密度函数
    在这个区域上的积分。当概率密度函数存在的时候，累积分布函数是概率密度函数的积分

    f(x)是概率密度函数，定义是取值落在某个区域之内的概率则为概率密度函数在这个区域上的积分

    :param cut_num: 切片为几个区间
    :param ndarry: ndarry: [0, 1] 的随机变量值数组
    :return: 构建的分布函数 pdf ndarry
    """
    if ndarry is None or ndarry.size == 0:
        print('Array can not be empty')
        return None

    arr_step = np.linspace(start=0, stop=1, num=cut_num)
    pdf_arr = []
    for i in range(cut_num - 1):
        pdf_arr.append(
            np.count_nonzero(np.logical_and(ndarry > arr_step[i], ndarry < arr_step[i + 1])) / ndarry.size)

    print(pdf_arr)
    return np.array(pdf_arr)


def plot_pdf(ndarry: np.ndarray, cut_num: int):
    """
    绘制 cdf 为柱状图
    :param ndarry: pdf ndarry
    :param cut_num: 切片为几个区间
    :return: None
    """
    if ndarry is None or ndarry.size == 0:
        print('Array can not be empty')
        return None

    x = np.linspace(start=0, stop=1, num=cut_num - 1)
    plt.figure(figsize=(32, 16), dpi=160)
    plt.plot(x, ndarry)
    plt.ylim(np.min(ndarry) - 0.015, np.max(ndarry) + 0.015)
    plt.title('Probability density function    Interval number=' + ndarry.size.__str__(),
              fontdict={'family': FONT, 'size': TITLE_FONT_SIZE})
    plt.xticks(x)
    plt.xticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    plt.yticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    plt.xlabel('x - Значение аргумента', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.ylabel('f(x) - Вероятность', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.grid(linestyle='--', alpha=0.5)

    plt.savefig('./result/pdf_' + (ndarry.size + 1).__str__() + '.png')
    plt.show()


if __name__ == '__main__':
    for n in [10, 100, 1000, 10000]:
        random_array = np.random.uniform(low=0, high=1, size=n)
        result = math_indicators(random_array)

        print('-' * 50)
        print('n = ' + n.__str__() + '\n'
              + 'M   = ' + result[0].__str__() + '\n'
              + 'M_S = ' + result[1].__str__() + '\n'
              + 'D   = ' + result[2].__str__() + '\n'
              + 'D_S = ' + result[3].__str__() + '\n\n')

        arr_auto = autocorrelation(random_array)
        plot_autocorrelation(arr_auto)

        if n == 10000:
            arr_cdf = get_cdf(random_array)
            plot_cdf(arr_cdf)

            arr_pdf = get_pdf(random_array, cut_num=20)
            plot_pdf(arr_pdf, cut_num=20)
    pass
