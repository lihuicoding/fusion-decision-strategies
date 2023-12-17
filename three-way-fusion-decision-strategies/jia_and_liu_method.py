import numpy as np
import three_way_main_procedure


def jia_and_liu_method(data, weight_vector, theta):
    """
    JiaFan和LiuPeiDe相对损失函数方法，A novel three-way decision model under multiple-criteria environment
    :param data:
    :param theta:
    :param weight_vector:
    :return: 三种排序结果，alpha,beta,gamma
    """
    n, m = data.shape
    data_max = data.max(axis=0)
    data_min = data.min(axis=0)
    theta = np.array([theta for i in range(m)])
    alpha, beta, gamma = np.zeros(n), np.zeros(n), np.zeros(n)
    # 计算阈值
    for i in range(n):
        alpha[i] = np.sum(weight_vector * (1 - theta) * (data_max - data[i])) / (
                np.sum(weight_vector * (1 - theta) * (data_max - data[i])) + np.sum(
            weight_vector * theta * (data[i] - data_min)))
        beta[i] = np.sum(weight_vector * theta * (data_max - data[i])) / (
                np.sum(weight_vector * theta * (data_max - data[i])) + np.sum(
            weight_vector * (1 - theta) * (data[i] - data_min)))
        gamma[i] = np.sum(weight_vector * (data_max - data[i]) / (data_max - data_min))

    rank_by_alpha = np.argsort(alpha)
    rank_by_beta = np.argsort(beta)
    rank_by_gamma = np.argsort(gamma)

    return [rank_by_alpha + 1, rank_by_beta + 1, rank_by_gamma + 1]


if __name__ == "__main__":
    file = r'dataset/example.csv'
    data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(6)))
    data_normalized = three_way_main_procedure.data_normalize(data, [0, 1, 1, 1, 1, 1])
    print("标准化后的数据:\n", data_normalized)
    weight_vector = np.array([0.2, 0.25, 0.30, 0.14, 0.03, 0.08])
    theta = 0.29
    rank_result = jia_and_liu_method(data_normalized, weight_vector, theta)
    print("alpha排序结果:\n", rank_result[0])
    print("beta排序结果:\n", rank_result[1])
    print("gamma排序结果:\n", rank_result[2])
