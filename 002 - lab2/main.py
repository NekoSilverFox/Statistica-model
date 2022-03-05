# ------*------ coding: utf-8 ------*------
# @Time    : 2022/2/19 19:15
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : Statistica-model
# @File    : main.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
import st_method
import numpy as np
import irandom


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

    cut_num = 100
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


if __name__ == '__main__':
    # # 【2.1】РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ (дискретное)
    # uniform_distribution()
    #
    # # 【2.2】БИНОМИАЛЬНОЕ РАСПРЕДЕЛЕНИЕ
    # binomial_distribution()
    #
    # # 【2.3.1】ГЕОМЕТРИЧЕСКОЕ РАСПРЕДЕЛЕНИЕ (Алгоритм 1)
    # geometric_distribution(arr_geometric=irandom.irngeo_1(p=0.5, size=10000),
    #                        img_save_fold='./result/【2.3.1】Geometric distribution/')
    #
    # # 【2.3.2】ГЕОМЕТРИЧЕСКОЕ РАСПРЕДЕЛЕНИЕ (Алгоритм 2)
    # geometric_distribution(arr_geometric=irandom.irngeo_2(p=0.5, size=10000),
    #                        img_save_fold='./result/【2.3.2】Geometric distribution/')
    #
    # # 【2.3.3】ГЕОМЕТРИЧЕСКОЕ РАСПРЕДЕЛЕНИЕ (Алгоритм 3)
    geometric_distribution(arr_geometric=irandom.irngeo_3(p=0.5, size=10000),
                           img_save_fold='./result/【2.3.3】Geometric distribution/')
    #
    # # 【2.4.1】РАСПРЕДЕЛЕНИЕ ПУАССОНА (Алгоритм 1)
    # poisson_distribution(arr_poisson=irandom.irnpoi(mu=10, size=10000),
    #                      img_save_fold='./result/【2.4.1】Poisson distribution/')
    #
    # # 【2.4.2】РАСПРЕДЕЛЕНИЕ ПУАССОНА (Алгоритм 2)
    poisson_distribution(arr_poisson=irandom.irnpsn(mu=10, size=10000),
                         img_save_fold='./result/【2.4.2】Poisson distribution/')

    pass
