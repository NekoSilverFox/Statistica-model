# ------*------ coding: utf-8 ------*------
# @Time    : 2022/2/19 19:15
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Statistica-model
# @File    : main.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
import math

import st_method
import numpy as np
import irandom
import matplotlib.pyplot as plt
import pandas as pd


TITLE_FONT_SIZE = 36
LABEL_FONT_SIZE = 36
TICK_FONT_SIZE = 32
FONT = 'Times New Roman'


def uniform_distribution():
    """
    CN：均匀分布
    EN：Uniform distribution
    RU：Равномерное распределение(2.1)
    :return: None
    """
    arr_uniform = irandom.irnuni(low=1, high=100, size=10000)
    M = arr_uniform.mean()
    D = arr_uniform.var()

    print('>' * 50, '\n',
          '【2.1】Равномерное распределение:\n',
          '\tM= ', M, '\n',
          '\tD= ', D, '\n')

    arr_cdf = st_method.get_cdf(arr_uniform)
    st_method.plot_cdf(cdf_ndarry=arr_cdf,
                       kind='scatter',
                       x_tick_min=1,
                       x_tick_max=100,
                       save_path='./result/【2.1】Uniform distribution/cdf_' + arr_cdf.size.__str__() + '.png')

    cut_num = 101
    arr_pdf = st_method.get_pdf(arr_uniform, cut_num=cut_num)
    st_method.plot_pdf(pdf_ndarry=arr_pdf,
                       kind='scatter',
                       x_tick_min=1,
                       x_tick_max=100,
                       y_tick_min=0,
                       y_tick_max=arr_pdf.max() + 0.02,
                       cut_num=cut_num,
                       save_path='./result/【2.1】Uniform distribution/pdf_' + (arr_pdf.size + 1).__str__() + '.png')

    print('-' * 50, '\n\n')
    return


def binomial_distribution():
    """
    CN：二项式分布
    EN：Binomial distribution
    RU：Биномиальное распределение(2.2)
    :return: None
    """
    arr_binomial = irandom.irnbnl(n=10, p=0.5, size=10000)
    M = arr_binomial.mean()
    D = arr_binomial.var()

    print('>' * 50, '\n',
          '【2.2】Биномиальное распределение:\n',
          '\tM= ', M, '\n',
          '\tD= ', D, '\n')

    arr_cdf = st_method.get_cdf(arr_binomial)
    st_method.plot_cdf(cdf_ndarry=arr_cdf,
                       kind='plot',
                       x_tick_min=1,
                       x_tick_max=10,
                       save_path='./result/【2.2】Binomial distribution/cdf_' + arr_cdf.size.__str__() + '.png')

    cut_num = 10
    arr_pdf = st_method.get_pdf(arr_binomial, cut_num=cut_num)
    st_method.plot_pdf(pdf_ndarry=arr_pdf,
                       kind='plot',
                       x_tick_min=1,
                       x_tick_max=10,
                       y_tick_min=0,
                       y_tick_max=arr_pdf.max() + 0.1,
                       cut_num=cut_num,
                       save_path='./result/【2.2】Binomial distribution/pdf_' + (arr_pdf.size + 1).__str__() + '.png')

    print('-' * 50, '\n')
    return


def geometric_distribution(arr_geometric: np.ndarray, img_save_fold: str):
    """
    CN：几何分布
    EN：Geometric distribution
    RU：Геометрическое распределение
    :param arr_geometric: 具有几何分布的 numpy.ndarry
    :param img_save_fold: 图形存储的【文件夹】
    :return: None
    """
    M = arr_geometric.mean()
    D = arr_geometric.var()

    print('>' * 50, '\n',
          '【2.3】Геометрическое распределение:\n',
          '\tM= ', M, '\n',
          '\tD= ', D, '\n')

    arr_cdf = st_method.get_cdf(arr_geometric)
    st_method.plot_cdf(cdf_ndarry=arr_cdf,
                       kind='plot',
                       x_tick_min=1,
                       x_tick_max=arr_geometric.max(),
                       save_path=img_save_fold + 'cdf_' + arr_cdf.size.__str__() + '.png')

    cut_num = 14
    arr_pdf = st_method.get_pdf(arr_geometric, cut_num=cut_num)
    st_method.plot_pdf(pdf_ndarry=arr_pdf,
                       kind='plot',
                       x_tick_min=1,
                       x_tick_max=cut_num - 1,
                       y_tick_min=0,
                       y_tick_max=arr_pdf.max() + 0.1,
                       cut_num=cut_num,
                       save_path=img_save_fold + 'pdf_' + (arr_pdf.size + 1).__str__() + '.png')

    print('-' * 50, '\n')
    return


def poisson_distribution(arr_poisson: np.ndarray, img_save_fold: str):
    """
    CN：泊松分布
    EN：Poisson distribution
    RU：Распределение Пуассона
    :param arr_poisson: 泊松分布的 numpy.ndarry 数组
    :param img_save_fold: 图形存储的【文件夹】
    :return: None
    """
    M = arr_poisson.mean()
    D = arr_poisson.var()

    print('>' * 50, '\n',
          '【2.4】Распределение Пуассона:\n',
          '\tM= ', M, '\n',
          '\tD= ', D, '\n')

    arr_cdf = st_method.get_cdf(arr_poisson)
    st_method.plot_cdf(cdf_ndarry=arr_cdf,
                       kind='plot',
                       x_tick_min=1,
                       x_tick_max=arr_poisson.max(),
                       save_path=img_save_fold + 'cdf_' + arr_cdf.size.__str__() + '.png')

    cut_num = 14
    arr_pdf = st_method.get_pdf(arr_poisson, cut_num=cut_num)
    st_method.plot_pdf(pdf_ndarry=arr_pdf,
                       kind='plot',
                       x_tick_min=1,
                       x_tick_max=cut_num - 1,
                       y_tick_min=0,
                       y_tick_max=arr_pdf.max() + 0.1,
                       cut_num=cut_num,
                       save_path=img_save_fold + 'pdf_' + (arr_pdf.size + 1).__str__() + '.png')

    print('-' * 50, '\n')
    return


def normal_distribution(arr_normal: np.ndarray, img_save_fold: str):
    """
    CN：正态分布
    EN：Normal distribution
    RU：Нормальное распределение
    :param arr_normal: 具有正态分布的 numpy.ndarry
    :param img_save_fold: 图形存储的【文件夹】
    :return: None
    """
    M = arr_normal.mean()
    D = arr_normal.var()

    print('>' * 50, '\n',
          '【2.2】Нормальное распределение:\n',
          '\tM= ', M, '\n',
          '\tD= ', D, '\n')

    arr_cdf = st_method.get_cdf(arr_normal)
    st_method.plot_cdf(cdf_ndarry=arr_cdf,
                       kind='plot',
                       x_tick_min=1,
                       x_tick_max=arr_normal.max(),
                       save_path=img_save_fold + 'cdf_' + arr_cdf.size.__str__() + '.png')

    cut_num = 21
    arr_pdf = st_method.get_pdf(arr_normal, cut_num=cut_num)
    st_method.plot_pdf(pdf_ndarry=arr_pdf,
                       kind='plot',
                       x_tick_min=arr_normal.min(),
                       x_tick_max=arr_normal.max(),
                       y_tick_min=0,
                       y_tick_max=arr_pdf.max() + 0.1,
                       cut_num=cut_num,
                       save_path=img_save_fold + 'pdf_' + (arr_pdf.size + 1).__str__() + '.png')

    print('-' * 50, '\n')
    return


def exponential_distribution(arr_exp: np.ndarray, img_save_fold: str):
    """
    CN：指数分布
    EN：Exponential distribution
    RU：Экспоненциальное распределение
    :param arr_exp: 具有正态分布的 numpy.ndarry
    :param img_save_fold: 图形存储的【文件夹】
    :return: None
    """
    M = arr_exp.mean()
    D = arr_exp.var()

    print('>' * 50, '\n',
          '【2.3】Экспоненциальное распределение:\n',
          '\tM= ', M, '\n',
          '\tD= ', D, '\n')

    arr_cdf = st_method.get_cdf(arr_exp)
    st_method.plot_cdf(cdf_ndarry=arr_cdf,
                       kind='plot',
                       x_tick_min=0,
                       # x_tick_max=arr_exp.max(),
                       x_tick_max=5,
                       save_path=img_save_fold + 'cdf_' + arr_cdf.size.__str__() + '.png')

    cut_num = 21
    arr_pdf = st_method.get_pdf(arr_exp, cut_num=cut_num)
    st_method.plot_pdf(pdf_ndarry=arr_pdf,
                       kind='plot',
                       x_tick_min=arr_exp.min(),
                       x_tick_max=arr_exp.max(),
                       y_tick_min=0,
                       y_tick_max=arr_pdf.max() + 0.1,
                       cut_num=cut_num,
                       save_path=img_save_fold + 'pdf_' + (arr_pdf.size + 1).__str__() + '.png')

    print('-' * 50, '\n')
    return


def chi_square_distribution(arr_exp: np.ndarray, img_save_fold: str):
    """
    CN：卡方分布
    EN：chi-square distribution
    RU：Хи-Квадрат Распределение
    :param arr_exp: 具有卡方态分布的 numpy.ndarry
    :param img_save_fold: 图形存储的【文件夹】
    :return: None
    """
    M = arr_exp.mean()
    D = arr_exp.var()

    print('>' * 50, '\n',
          '【2.4】Хи-Квадрат Распределение:\n',
          '\tM= ', M, '\n',
          '\tD= ', D, '\n')

    arr_cdf = st_method.get_cdf(arr_exp)
    st_method.plot_cdf(cdf_ndarry=arr_cdf,
                       kind='plot',
                       x_tick_min=0,
                       x_tick_max=arr_exp.max(),
                       save_path=img_save_fold + 'cdf_' + arr_cdf.size.__str__() + '.png')

    cut_num = 21
    arr_pdf = st_method.get_pdf(arr_exp, cut_num=cut_num)
    st_method.plot_pdf(pdf_ndarry=arr_pdf,
                       kind='plot',
                       x_tick_min=arr_exp.min(),
                       x_tick_max=arr_exp.max(),
                       y_tick_min=0,
                       y_tick_max=arr_pdf.max() + 0.1,
                       cut_num=cut_num,
                       save_path=img_save_fold + 'pdf_' + (arr_pdf.size + 1).__str__() + '.png')

    print('-' * 50, '\n')
    return


def student_t_distribution(arr_exp: np.ndarray, img_save_fold: str):
    """
    CN：学生t-分布
    EN：Student's t-distribution
    RU：Распределение Стьюдента
    :param arr_exp: 具有学生t-分布的 numpy.ndarry
    :param img_save_fold: 图形存储的【文件夹】
    :return: None
    """
    M = arr_exp.mean()
    D = arr_exp.var()

    print('>' * 50, '\n',
          '【2.5】Распределение Стьюдента:\n',
          '\tM= ', M, '\n',
          '\tD= ', D, '\n')

    arr_cdf = st_method.get_cdf(arr_exp)
    st_method.plot_cdf(cdf_ndarry=arr_cdf,
                       kind='plot',
                       x_tick_min=0,
                       x_tick_max=arr_exp.max(),
                       save_path=img_save_fold + 'cdf_' + arr_cdf.size.__str__() + '.png')

    cut_num = 21
    arr_pdf = st_method.get_pdf(arr_exp, cut_num=cut_num)
    st_method.plot_pdf(pdf_ndarry=arr_pdf,
                       kind='plot',
                       x_tick_min=arr_exp.min(),
                       x_tick_max=arr_exp.max(),
                       y_tick_min=0,
                       y_tick_max=arr_pdf.max() + 0.1,
                       cut_num=cut_num,
                       save_path=img_save_fold + 'pdf_' + (arr_pdf.size + 1).__str__() + '.png')

    print('-' * 50, '\n')
    return


def weibull_distribution(arr_weibull: np.ndarray, img_save_fold: str):
    """
    CN: 韦伯分布
    EN: Weibull distribution
    RU: Распределение Вейбулла
    :param arr_weibull: 具有韦伯分布的 numpy.ndarry
    :param img_save_fold: 图形存储的【文件夹】
    :return: None
    """
    M = arr_weibull.mean()
    D = arr_weibull.var()

    print('>' * 50, '\n',
          '【2.6】Распределение Вейбулла:\n',
          '\tM= ', M, '\n',
          '\tD= ', D, '\n')

    arr_cdf = st_method.get_cdf(arr_weibull)
    st_method.plot_cdf(cdf_ndarry=arr_cdf,
                       kind='plot',
                       x_tick_min=0,
                       x_tick_max=arr_weibull.max(),
                       save_path=img_save_fold + 'cdf_' + arr_cdf.size.__str__() + '.png')

    cut_num = 21
    arr_pdf = st_method.get_pdf(arr_weibull, cut_num=cut_num)
    st_method.plot_pdf(pdf_ndarry=arr_pdf,
                       kind='plot',
                       x_tick_min=arr_weibull.min(),
                       x_tick_max=arr_weibull.max(),
                       y_tick_min=0,
                       y_tick_max=arr_pdf.max() + 0.1,
                       cut_num=cut_num,
                       save_path=img_save_fold + 'pdf_' + (arr_pdf.size + 1).__str__() + '.png')

    print('-' * 50, '\n')
    return


def plot_weibull_hist(arr_weibull: np.ndarray, save_path: str):
    """
    绘制直方图并保存
    :param arr_weibull: 具有韦伯分布的数组
    :param save_path: 图像存储路径
    :return:
    """
    plt.figure(figsize=(18, 10), dpi=100)
    plt.hist(x=arr_weibull, bins=10, density=True)
    plt.grid(linestyle='--', alpha=0.5)
    plt.title('Распределение Вейбулла \n'
              'Section=' + arr_weibull.max().__str__(),
              fontdict={'family': FONT, 'size': TITLE_FONT_SIZE})
    plt.xlabel('x', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.ylabel('P(x)', fontdict={'family': FONT, 'size': LABEL_FONT_SIZE})
    plt.xticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    plt.yticks(fontproperties=FONT, size=TICK_FONT_SIZE)
    # plt.xlim(0, 15)
    plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    # 【2.1】РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ (дискретное)
    # uniform_distribution()

    # 【2.2.1】Нормальное распределение
    # normal_distribution(irandom.irnnrm_1(size=10000), './result/【2.2.1】Нормальное распределение/')

    # 【2.2.2】Нормальное распределение
    # normal_distribution(irandom.irnnrm_1(size=10000), './result/【2.2.2】Нормальное распределение/')

    # 【2.3】Экспоненциальное распределение
    # exponential_distribution(irandom.irnexp(beta=1, size=10000), './result/【2.3】Экспоненциальное распределение/')

    # 【2.4】Хи-Квадрат Распределение
    # chi_square_distribution(irandom.irnchis(n=10, size=10000), './result/【2.4】Хи-Квадрат Распределение/')

    # 【2.5】Распределение Стьюдента
    # student_t_distribution(irandom.irnstud(n=10, size=10000), './result/【2.5】Распределение Стьюдента/')

    # 【2.6】lab3+ Распределение Вейбулла
    # arr_weibull = irandom.irnweibull(k=5, l=1, size=100)  # 获取随机数组
    # arr_weibull_exp_cdf = irandom.weibull_exp_cdf(k=5, l=1, arr_exp=arr_weibull)  # 根据随机数组计算累积概率密度
    # arr_obs = np.linspace(start=0.01, stop=1, num=100)
    # df_weibull = pd.concat([pd.Series(np.sort(arr_weibull)),
    #                         pd.Series(arr_obs),
    #                         pd.Series(arr_weibull_exp_cdf),
    #                         pd.Series(arr_obs - arr_weibull_exp_cdf)], axis=1)
    # df_weibull.index = np.linspace(start=1, stop=100, num=100, dtype=np.int64)
    # df_weibull.columns = ['random_weibull', 'F(obs)', 'F(exp)', 'F(obs)-F(exp)']
    # df_weibull.to_csv(path_or_buf='./result/K-S test for Weibull.csv')
    # print(df_weibull, '\n',
    #       '-*-' * 20, '\n',
    #       'D_n MAX = ', df_weibull['F(obs)-F(exp)'].max())
    # weibull_distribution(arr_weibull, './result/【2.6】Распределение Вейбулла/')
    # plot_weibull_hist(arr_weibull=arr_weibull,
    #                   save_path='./result/【2.6】Распределение Вейбулла/hist_weibull.png')


    # 【2.6】lab3+ Распределение Рэлея
    arr_rayleigh = irandom.irnrayleigh(mu=1.0, size=100)
    arr_rayleigh_exp_cdf = irandom.rayleigh_exp_cdf(mu=1.0, arr_exp=arr_rayleigh)  # 根据随机数组计算累积概率密度
    arr_obs = np.linspace(start=0.01, stop=1, num=100)
    df_rayleigh = pd.concat([pd.Series(np.sort(arr_rayleigh)),
                             pd.Series(arr_obs),
                             pd.Series(arr_rayleigh_exp_cdf),
                             pd.Series(arr_obs - arr_rayleigh_exp_cdf)], axis=1)
    df_rayleigh.index = np.linspace(start=1, stop=100, num=100, dtype=np.int64)
    df_rayleigh.columns = ['random_rayleigh', 'F(obs)', 'F(exp)', 'F(obs)-F(exp)']
    df_rayleigh.to_csv(path_or_buf='./result/K-S test for Rayleigh.csv')
    print(df_rayleigh, '\n',
          '-*-' * 20, '\n',
          'D_n MAX = ', df_rayleigh['F(obs)-F(exp)'].max(), '\n',
          '临界值 = ', 1.36 / math.sqrt(100))
    # weibull_distribution(arr_rayleigh, './result/【2.6】Распределения Релея/')
    # plot_weibull_hist(arr_weibull=arr_rayleigh,
    #                   save_path='./result/【2.6】Распределения Релея/hist_weibull.png')



    pass
