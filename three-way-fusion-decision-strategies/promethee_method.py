import numpy as np
import three_way_main_procedure

# 函数用于计算PROMETHEE流的流出和流入
def calculate_phis(evaluation_matrix, weights):
    num_alternatives, num_criteria = evaluation_matrix.shape
    phis_plus = np.zeros((num_alternatives, num_alternatives))
    phis_minus = np.zeros((num_alternatives, num_alternatives))

    for i in range(num_alternatives):
        for j in range(num_alternatives):
            if i != j:
                for k in range(num_criteria):
                    diff = evaluation_matrix[i, k] - evaluation_matrix[j, k]
                    if diff > 0:
                        phis_plus[i, j] += weights[k]
                    elif diff < 0:
                        phis_minus[i, j] += weights[k]

    return phis_plus, phis_minus


def promethee_method(evaluation_matrix, weights):
    # 计算流出和流入
    phis_plus, phis_minus = calculate_phis(evaluation_matrix, weights)

    # 计算总流出和总流入
    total_phis_plus = np.sum(phis_plus, axis=1)
    total_phis_minus = np.sum(phis_minus, axis=1)

    # 计算PROMETHEE综合得分
    net_flows = total_phis_plus - total_phis_minus
    ranking = np.argsort(net_flows)[::-1]  # 从大到小排序

    # print("PROMETHEE综合得分：", net_flows)
    # print("排序结果：", ranking)
    return ranking+1


if __name__ == "__main__":
    # # 评价矩阵，每行表示一个决策方案，每列表示一个评价标准
    # evaluation_matrix = np.array([
    #     [4, 8],
    #     [6, 5],
    #     [9, 3]
    # ])
    # # 权重向量，表示评价标准的权重
    # weights = np.array([0.6, 0.4])
    file = r'dataset/example.csv'
    data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(6)))
    data_normalized = three_way_main_procedure.data_normalize(data, [0, 1, 1, 1, 1, 1])
    weight_vector = np.array([0.15, 0.2, 0.1, 0.25, 0.1, 0.2])
    print(promethee_method(data_normalized, weight_vector))