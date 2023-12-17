import three_way_main_procedure
import numpy as np


def topsis(data_normalized, weight_vector):
    """
    topsis排序方法
    :param data_normalized: 已经标准化之后的数据
    :param weight_vector: 属性权重np.array
    :return: 排序结果
    """
    data_max = data_normalized.max(axis=0)  # 正理想解
    data_min = data_normalized.min(axis=0)  # 负理想解
    dis_pos = np.sqrt(((data_normalized - data_max) ** 2 * weight_vector).sum(axis=-1))
    dis_neg = np.sqrt(((data_normalized - data_min) ** 2 * weight_vector).sum(axis=-1))
    result = dis_neg / (dis_pos + dis_neg)
    return np.argsort(result)[::-1] + 1


if __name__ == '__main__':
    file = r'dataset/example.csv'
    data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(6)))
    data_normalized = three_way_main_procedure.data_normalize(data, [0, 1, 1, 1, 1, 1])
    weight_vector = np.array([0.2, 0.25, 0.30, 0.14, 0.03, 0.08])
    print(topsis(data_normalized, weight_vector))
