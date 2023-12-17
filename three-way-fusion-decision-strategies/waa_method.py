import three_way_main_procedure
import numpy as np


def waa(data_normalized, weight_vector):
    """
    waa加权平均排序方法
    :param data_normalized: 已经标准化之后的数据
    :param weight_vector: 属性权重np.array
    :return: 排序结果
    """
    n, m = data_normalized.shape
    # 加权平均值
    weighted_values = np.sum(data_normalized * np.tile(weight_vector, (n, 1)), axis=-1)
    return np.argsort(weighted_values)[::-1] + 1


if __name__ == '__main__':
    file = r'dataset/example.csv'
    data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(6)))
    data_normalized = three_way_main_procedure.data_normalize(data, [0, 1, 1, 1, 1, 1])
    weight_vector = np.array([0.2, 0.25, 0.30, 0.14, 0.03, 0.08])
    print(waa(data_normalized, weight_vector))
