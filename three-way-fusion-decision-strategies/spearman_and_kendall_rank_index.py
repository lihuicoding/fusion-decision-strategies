import pandas as pd


def spearman_and_kendall_rank_index(x, y):
    """
    计算两个变量之间的斯皮尔曼相关系数和肯德尔相关系数
    :param x:
    :param y:
    :return: SRCC和KRCC
    """
    x_index = pd.Series(x).sort_values().index
    y_index = pd.Series(y).sort_values().index
    # print(x_index)
    # print(y_index)
    srcc = float(format(pd.Series(x_index).corr(pd.Series(y_index), method='spearman'), '.4f'))
    krcc = float(format(pd.Series(x_index).corr(pd.Series(y_index), method='kendall'), '.4f'))
    return srcc, krcc


if __name__ == '__main__':
    x = [7, 2, 1, 6, 4, 3, 5, 8]
    y = [7, 2, 1, 6, 3, 4, 5, 8]
    # x = [6, 8, 3, 1, 5, 7, 4, 2]
    # y = [6, 8, 3, 1, 2, 5, 4, 7]
    # x = [2, 3, 7, 4, 5, 6, 1, 8]
    # y = [2, 3, 8, 4, 5, 6, 1, 7]
    # x = [7, 1, 2, 4, 5, 6, 3, 8]
    # y = [7, 1, 2, 4, 5, 6, 8, 3]
    print(spearman_and_kendall_rank_index(x, y))
