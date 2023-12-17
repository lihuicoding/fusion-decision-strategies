import numpy as np
import pandas as pd
import three_way_main_procedure
import waa_method
import topsis_method
import jia_and_liu_method
import zhang_tfs_method


def inherent_rank_index(data, rank_result):
    """
    计算新的排序结果与原始数据内在的固有排序之间的一致性
    :param data: 正规化之后的数据
    :param rank_result: 被检验方法的排序结果
    :return: 一致性指数
    """
    # 定义一个可比矩阵
    n, m = data.shape
    compare_matrix = np.zeros((n, n), dtype=bool)
    compared_count, inverse_count = 0, 0
    # 获取对象的排序索引
    rank_index = pd.Series(rank_result).sort_values().index
    # 统计可比对的数量和逆序对的数量
    for i in range(n):
        for j in range(n):
            # 如果在全部属性上都大于说明是可比的
            compare_matrix[i, j] = np.all(data[i, :] > data[j, :])
            if compare_matrix[i, j]:
                compared_count += 1
                if rank_index[i] > rank_index[j]:  # 如果i排在了j的后面，那么就是逆序排序了，说明排序错误
                    inverse_count += 1
    if compared_count == 0:
        return 1
    return 1 - inverse_count / compared_count  # 正确排序率


if __name__ == '__main__':
    print("tripadvisor_review:")
    file = r'dataset/tripadvisor_review.csv'
    data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(1, 11)))
    data_normalized = three_way_main_procedure.data_normalize(data, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # print("标准化后的数据:\n", data_normalized)
    n, m = data_normalized.shape
    concepts = np.random.random((16, 10))
    # print("concepts:\n", concepts)
    weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
    expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
    theta_vector = np.ones(concepts.shape[0]) * 0.45
    theta = 0.45
    aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

    # our_method
    our_method_classify_result, our_method_rank_result = three_way_main_procedure.three_way_model(data_normalized,
                                                                                                  concepts,
                                                                                                  theta_vector,
                                                                                                  weight_vector,
                                                                                                  expecter_weight_vector)

    # waa_method
    waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

    # topsis_method
    topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

    # jia_and_liu_method
    jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

    # zhang_tfs_method
    zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                     aggregated_concept)

    print("our_method正确排序率：", inherent_rank_index(data_normalized, our_method_rank_result))
    print("waa_method正确排序率：", inherent_rank_index(data_normalized, waa_method_rank_result))
    print("topsis_method正确排序率：", inherent_rank_index(data_normalized, topsis_method_rank_result))
    print("jia_and_liu_method正确排序率：", inherent_rank_index(data_normalized, jia_and_liu_method_rank_result[0]))
    print("zhang_tfs_absolute_method正确排序率：", inherent_rank_index(data_normalized, zhang_tfs_method_rank_result[0]))
    print("zhang_tfs_relative_method正确排序率：", inherent_rank_index(data_normalized, zhang_tfs_method_rank_result[1]))
    print("-----------------------------------------------------------------------------------------------------------")

    # computer_hardware
    print("computer_hardware:")
    file = r'dataset/computer_hardware.csv'
    data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(2, 8)))
    data_normalized = three_way_main_procedure.data_normalize(data, [0, 1, 1, 1, 1, 1])
    # print("标准化后的数据:\n", data_normalized)
    n, m = data_normalized.shape
    concepts = np.random.random((16, 6))
    # print("concepts:\n", concepts)
    weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
    expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
    theta_vector = np.ones(concepts.shape[0]) * 0.45
    theta = 0.45
    aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

    # our_method
    our_method_classify_result, our_method_rank_result = three_way_main_procedure.three_way_model(data_normalized,
                                                                                                  concepts,
                                                                                                  theta_vector,
                                                                                                  weight_vector,
                                                                                                  expecter_weight_vector)

    # waa_method
    waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

    # topsis_method
    topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

    # jia_and_liu_method
    jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

    # zhang_tfs_method
    zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                     aggregated_concept)

    print("our_method正确排序率：", inherent_rank_index(data_normalized, our_method_rank_result))
    print("waa_method正确排序率：", inherent_rank_index(data_normalized, waa_method_rank_result))
    print("topsis_method正确排序率：", inherent_rank_index(data_normalized, topsis_method_rank_result))
    print("jia_and_liu_method正确排序率：", inherent_rank_index(data_normalized, jia_and_liu_method_rank_result[0]))
    print("zhang_tfs_absolute_method正确排序率：", inherent_rank_index(data_normalized, zhang_tfs_method_rank_result[0]))
    print("zhang_tfs_relative_method正确排序率：", inherent_rank_index(data_normalized, zhang_tfs_method_rank_result[1]))
    print("-----------------------------------------------------------------------------------------------------------")

    # winequality_red
    print("winequality_red:")
    file = r'dataset/winequality_red.csv'
    data = three_way_main_procedure.read_csv_data(file, 1, (1, 2, 7, 8, 9, 10))
    data_normalized = three_way_main_procedure.data_normalize(data, [0, 1, 0, 0, 1, 1])
    # print("标准化后的数据:\n", data_normalized)
    n, m = data_normalized.shape
    concepts = np.random.random((16, 6))
    # print("concepts:\n", concepts)
    weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
    expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
    theta_vector = np.ones(concepts.shape[0]) * 0.45
    theta = 0.45
    aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

    # our_method
    our_method_classify_result, our_method_rank_result = three_way_main_procedure.three_way_model(data_normalized,
                                                                                                  concepts,
                                                                                                  theta_vector,
                                                                                                  weight_vector,
                                                                                                  expecter_weight_vector)

    # waa_method
    waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

    # topsis_method
    topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

    # jia_and_liu_method
    jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

    # zhang_tfs_method
    zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                     aggregated_concept)

    print("our_method正确排序率：", inherent_rank_index(data_normalized, our_method_rank_result))
    print("waa_method正确排序率：", inherent_rank_index(data_normalized, waa_method_rank_result))
    print("topsis_method正确排序率：", inherent_rank_index(data_normalized, topsis_method_rank_result))
    print("jia_and_liu_method正确排序率：", inherent_rank_index(data_normalized, jia_and_liu_method_rank_result[0]))
    print("zhang_tfs_absolute_method正确排序率：", inherent_rank_index(data_normalized, zhang_tfs_method_rank_result[0]))
    print("zhang_tfs_relative_method正确排序率：", inherent_rank_index(data_normalized, zhang_tfs_method_rank_result[1]))
    print("-----------------------------------------------------------------------------------------------------------")

    # winequality_white
    print("winequality_white:")
    file = r'dataset/winequality_white.csv'
    data = three_way_main_procedure.read_csv_data(file, 1, (7, 8, 4, 10))
    data_normalized = three_way_main_procedure.data_normalize(data, [0, 1, 0, 1])
    # print("标准化后的数据:\n", data_normalized)
    n, m = data_normalized.shape
    concepts = np.random.random((16, 4))
    # print("concepts:\n", concepts)
    weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
    expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
    theta_vector = np.ones(concepts.shape[0]) * 0.45
    theta = 0.45
    aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

    # our_method
    our_method_classify_result, our_method_rank_result = three_way_main_procedure.three_way_model(data_normalized,
                                                                                                  concepts,
                                                                                                  theta_vector,
                                                                                                  weight_vector,
                                                                                                  expecter_weight_vector)

    # waa_method
    waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

    # topsis_method
    topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

    # jia_and_liu_method
    jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

    # zhang_tfs_method
    zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                     aggregated_concept)

    print("our_method正确排序率：", inherent_rank_index(data_normalized, our_method_rank_result))
    print("waa_method正确排序率：", inherent_rank_index(data_normalized, waa_method_rank_result))
    print("topsis_method正确排序率：", inherent_rank_index(data_normalized, topsis_method_rank_result))
    print("jia_and_liu_method正确排序率：", inherent_rank_index(data_normalized, jia_and_liu_method_rank_result[0]))
    print("zhang_tfs_absolute_method正确排序率：", inherent_rank_index(data_normalized, zhang_tfs_method_rank_result[0]))
    print("zhang_tfs_relative_method正确排序率：", inherent_rank_index(data_normalized, zhang_tfs_method_rank_result[1]))
    print("-----------------------------------------------------------------------------------------------------------")