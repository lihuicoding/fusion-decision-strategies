import numpy as np
import pandas as pd


def local_ranking_index(X, Y):
    """
    计算局部排序指数
    :param X: 排序结果1
    :param Y: 排序结果2
    :return: 一致性指数
    """
    # 定义一个可比矩阵
    n = X.shape[0]  # 对象的个数
    x_index = pd.Series(X).sort_values().index  # 获取1-n个对象对应的排序索引
    y_index = pd.Series(Y).sort_values().index
    # print(x_index)
    # 比较矩阵
    number = 0  # 相交元素的个数
    for i in range(n):
        for j in range(n):
            if i != j:
                if x_index[i] > x_index[j] and y_index[i] < y_index[j]:
                    number += 1
    return 1 - 2 * number / (n * (n - 1))


if __name__ == "__main__":
    X = np.array([10, 9, 7, 6, 5, 4, 3, 8, 2, 1])
    # X = np.array([10, 7, 9, 6, 5, 4, 3, 8, 2, 1])
    Y = np.array([10, 6, 7, 5, 9, 4, 1, 2, 3, 8])
    print(local_ranking_index(X, Y))
