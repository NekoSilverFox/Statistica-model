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
TICK_FONT_SIZE = 32
FONT = 'Times New Roman'


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
    F(x)是分布函数，F(x)的取值表示变量小于 x 的概率是多少。比如说 F(x)=0.85，x=0.7，那么意思是随机变量小于 0.7 的概率为 0.85

    Ru：Эмпирическая интегральная функция распределения
    :param ndarry: 变量值数组
    :return: 构建的分布函数 cdf ndarry
    """
    if ndarry is None or ndarry.size == 0:
        print('Array can not be empty')
        return None

    arr_cdf = []
    arr_step = np.linspace(start=ndarry.min(), stop=ndarry.max(), num=100)
    for i in arr_step:
        arr_cdf.append(np.count_nonzero(ndarry < i) / ndarry.size)

    # print(arr_cdf)
    return np.array(arr_cdf)


def plot_cdf(cdf_ndarry: np.ndarray, kind: str, x_tick_min: int, x_tick_max: int, save_path: str):
    """
    绘制 cdf 为散点图
    :param cdf_ndarry:  cdf ndarry
    :param kind: 绘制图像的类型，可以为 plot | scatter | bar
    :param x_tick_min: x 轴坐标的最小值（应该最初数组的最小值）
    :param x_tick_max: x 轴坐标的最大值（应该最初数组的最大值）
    :param save_path: 图像存储路径
    :return: None
    """

    if cdf_ndarry is None or cdf_ndarry.size == 0:
        print('Array can not be empty')
        return None

    x = np.linspace(start=x_tick_min, stop=x_tick_max, num=100)
    plt.figure(figsize=(32, 16), dpi=160)

    if kind == 'plot':
        plt.plot(x, cdf_ndarry, linewidth=5, marker='o', markersize=30)
    elif kind == 'scatter':
        plt.scatter(x, cdf_ndarry, s=300)
    elif kind == 'bar':
        plt.bar(x, cdf_ndarry)
    else:
        print('`kind` should be plot | scatter | bar')
        return

    plt.title('Cumulative Distribution Function\n'
              'Number of sampling points=' + cdf_ndarry.size.__str__(),
              fontdict={'family': FONT, 'size': TITLE_FONT_SIZE})
    plt.xlabel('x - Значение аргумента', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.ylabel('F(x) - Вероятность', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.xticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    plt.yticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    plt.grid(linestyle='--', alpha=0.5)

    plt.savefig(save_path)
    plt.show()


def get_pdf(ndarry: np.ndarray, cut_num: int) -> np.ndarray:
    """ 获取概率密度函数的 ndarry
    在数学中，连续型随机变量的概率密度函数（Probability density function，简写作PDF [1]），
    在不致于混淆时可简称为密度函数，是一个描述这个随机变量的输出值，在某个确定的取值点附近的可能性的函数。
    图中，横轴为随机变量的取值，纵轴为概率密度函数的值，而随机变量的取值落在某个区域内的概率为概率密度函数
    在这个区域上的积分。当概率密度函数存在的时候，累积分布函数是概率密度函数的积分

    f(x)是概率密度函数，定义是取值落在某个区域之内的概率则为概率密度函数在这个区域上的积分

    :param cut_num: 切片为几个区间
    :param ndarry: ndarry: 变量值数组
    :return: 构建的分布函数 pdf ndarry
    """
    if ndarry is None or ndarry.size == 0:
        print('Array can not be empty')
        return None

    arr_pdf = []
    arr_step = np.linspace(start=ndarry.min(), stop=ndarry.max(), num=cut_num)
    for i in range(cut_num - 1):
        arr_pdf.append(
            np.count_nonzero(np.logical_and(ndarry > arr_step[i], ndarry < arr_step[i + 1])) / ndarry.size)

    print(arr_pdf)
    return np.array(arr_pdf)


def plot_pdf(pdf_ndarry: np.ndarray, kind: str, x_tick_min: int, x_tick_max: int,
             y_tick_min: int, y_tick_max: int, cut_num: int, save_path: str):
    """
    绘制 cdf 为图
    :param pdf_ndarry: pdf ndarry
    :param kind: 绘制图像的类型，可以为 plot | scatter | bar
    :param x_tick_min: x 轴坐标的最小值（应该最初数组的最小值）
    :param x_tick_max: x 轴坐标的最大值（应该最初数组的最大值）
    :param y_tick_min: y 轴坐标的最小值（应该最初数组的最小值）
    :param y_tick_max: y 轴坐标的最大值（应该最初数组的最大值）
    :param cut_num: 切片为几个区间
    :param save_path: 图像存储路径
    :return: None
    """

    if pdf_ndarry is None or pdf_ndarry.size == 0:
        print('Array can not be empty')
        return None

    x = np.linspace(start=x_tick_min, stop=x_tick_max, num=cut_num - 1)
    plt.figure(figsize=(32, 16), dpi=160)

    if kind == 'plot':
        plt.plot(x, pdf_ndarry, linewidth=5, marker='o', markersize=30)
    elif kind == 'scatter':
        plt.scatter(x, pdf_ndarry, s=300)
    elif kind == 'bar':
        plt.bar(x, pdf_ndarry)
    else:
        print('`kind` should be plot | scatter | bar')
        return

    plt.ylim(y_tick_min, y_tick_max)
    plt.title('Probability Density Function\nInterval number=' + pdf_ndarry.size.__str__(),
              fontdict={'family': FONT, 'size': TITLE_FONT_SIZE})
    plt.xticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    plt.yticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    plt.xlabel('x - Значение аргумента', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.ylabel('f(x) - Вероятность', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.grid(linestyle='--', alpha=0.5)

    plt.savefig(save_path)
    plt.show()
