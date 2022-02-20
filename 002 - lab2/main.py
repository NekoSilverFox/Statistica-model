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


def uniform_distribution():
    """
    CN：均匀分布
    EN：Uniform distribution
    RU：Равномерное распределение(2.1)
    :return: None
    """
    arr_uniform = np.random.uniform(low=1, high=100, size=10000)
    M = arr_uniform.mean()
    D = arr_uniform.var()

    print('>' * 50, '\n',
          'Равномерное распределение:\n',
          '\tM= ', M, '\n',
          '\tD= ', D, '\n')

    arr_cdf = st_method.get_cdf(arr_uniform)
    st_method.plot_cdf(cdf_ndarry=arr_cdf, x_tick_min=1, x_tick_max=100)



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
    arr_uniform = np.random.binomial(n=10, p=0.5, size=10000)
    M = arr_uniform.mean()
    D = arr_uniform.var()

    print('>' * 50, '\n',
          'Биномиальное распределение(2.2):\n',
          '\tM= ', M, '\n',
          '\tD= ', D, '\n')

    arr_cdf = st_method.get_cdf(arr_uniform)
    st_method.plot_cdf(cdf_ndarry=arr_cdf, x_tick_min=1, x_tick_max=10)

    cut_num = 10
    arr_pdf = st_method.get_pdf(arr_uniform, cut_num=cut_num)
    st_method.plot_pdf(pdf_ndarry=arr_pdf,
                       kind='plot',
                       x_tick_min=1,
                       x_tick_max=10,
                       y_tick_min=0,
                       y_tick_max=arr_pdf.max() + 0.1,
                       cut_num=cut_num,
                       save_path='./result/【2.2】Binomial distribution/pdf_'+ (arr_pdf.size + 1).__str__() + '.png')

    print('-' * 50, '\n')
    return




if __name__ == '__main__':
    uniform_distribution()
    binomial_distribution()
    pass
