import numpy as np
import three_way_main_procedure


def zhang_tfs_method(data, weight_vector, theta, concept):
    """
    zhang_kai TFS论文面向准则模糊概念的三支决策模型The Criterion-Oriented Three-Way Ranking and Clustering Strategies in Fuzzy Decision Environments
    :param data:
    :param weight_vector:
    :param theta:
    :param concept: 准则模糊概念
    :return:absolute排序结果，relative排序结果
    """
    n, m = data.shape

    # 计算损失函数
    loss_functions = []
    for i in range(n):
        PP = 0
        NP = np.sum(weight_vector * np.maximum(0, data[i] - concept))
        BP = theta * NP
        PN = np.sum(weight_vector * np.maximum(0, concept - data[i]))
        BN = theta * PN
        NN = 0
        loss_functions.append([[PP, PN], [BP, BN], [NP, NN]])
        # print(NP+PN)
    loss = np.array(loss_functions)

    # 计算阈值
    alpha = (loss[:, 0, 1] - loss[:, 1, 1]) / (loss[:, 0, 1] - loss[:, 1, 1] + loss[:, 1, 0] - loss[:, 0, 0])
    beta = (loss[:, 1, 1] - loss[:, 2, 1]) / (loss[:, 1, 1] - loss[:, 2, 1] + loss[:, 2, 0] - loss[:, 1, 0])

    # 计算条件概率
    fuzzy_lower_approximation, fuzzy_upper_approximation = np.zeros(n), np.zeros(n)
    for i in range(n):
        fuzzy_lower_approximation[i] = np.maximum(concept, 1 - data[i]).min()  # 绝对条件概率
        fuzzy_upper_approximation[i] = np.minimum(concept, data[i]).max()  # 相对条件概率
    absolute_probability = fuzzy_lower_approximation
    relative_probability = fuzzy_upper_approximation

    # absolute排序方法
    # 对象分类,where返回的是一个元组，所以取索引0
    pos = np.where(absolute_probability >= alpha)[0]  # 正域
    bnd = np.where((beta < absolute_probability) & (absolute_probability < alpha))[0]  # 边界域
    neg = np.where(absolute_probability <= beta)[0]  # 负域
    # 计算期望损失
    expect_loss = np.zeros((n, 3))
    for i in range(n):
        expect_loss[i, 0] = np.sum(loss[i, 0] * np.array([absolute_probability[i], 1 - absolute_probability[i]]))
        expect_loss[i, 1] = np.sum(loss[i, 1] * np.array([absolute_probability[i], 1 - absolute_probability[i]]))
        expect_loss[i, 2] = np.sum(loss[i, 2] * np.array([absolute_probability[i], 1 - absolute_probability[i]]))
    final_expect_loss = np.zeros(n)
    final_expect_loss[pos] = expect_loss[pos, 0]
    final_expect_loss[bnd] = expect_loss[bnd, 1]
    final_expect_loss[neg] = expect_loss[neg, 2]
    # 排序
    pos_rank = np.argsort(final_expect_loss[pos])  # 正域中的排序
    bnd_rank = np.argsort(final_expect_loss[bnd])  # 边界域中的排序
    neg_rank = np.argsort(final_expect_loss[neg])  # 负域中的排序
    absolute_rank_result = np.concatenate((pos[pos_rank], bnd[bnd_rank],
                                           neg[neg_rank]))  # 处理的时候都是从0开始，所以返回结果要加1,负域中结果相反
    # classify_result = (pos + 1, bnd + 1, neg + 1)

    # relative排序方法
    # 对象分类,where返回的是一个元组，所以取索引0
    pos = np.where(relative_probability >= alpha)[0]  # 正域
    bnd = np.where((beta < relative_probability) & (relative_probability < alpha))[0]  # 边界域
    neg = np.where(relative_probability <= beta)[0]  # 负域
    # 计算期望损失
    expect_loss = np.zeros((n, 3))
    for i in range(n):
        expect_loss[i, 0] = np.sum(loss[i, 0] * np.array([relative_probability[i], 1 - relative_probability[i]]))
        expect_loss[i, 1] = np.sum(loss[i, 1] * np.array([relative_probability[i], 1 - relative_probability[i]]))
        expect_loss[i, 2] = np.sum(loss[i, 2] * np.array([relative_probability[i], 1 - relative_probability[i]]))
    final_expect_loss = np.zeros(n)
    final_expect_loss[pos] = expect_loss[pos, 0]
    final_expect_loss[bnd] = expect_loss[bnd, 1]
    final_expect_loss[neg] = expect_loss[neg, 2]
    # 排序
    pos_rank = np.argsort(final_expect_loss[pos])  # 正域中的排序
    bnd_rank = np.argsort(final_expect_loss[bnd])  # 边界域中的排序
    neg_rank = np.argsort(final_expect_loss[neg])  # 负域中的排序
    relative_rank_result = np.concatenate((pos[pos_rank], bnd[bnd_rank],
                                           neg[neg_rank]))  # 处理的时候都是从0开始，所以返回结果要加1,负域中结果相反
    # classify_result = (pos + 1, bnd + 1, neg + 1)

    return [absolute_rank_result + 1, relative_rank_result + 1]


if __name__ == "__main__":
    file = r'dataset/example.csv'
    data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(6)))
    data_normalized = three_way_main_procedure.data_normalize(data, [0, 1, 1, 1, 1, 1])
    print("标准化后的数据:\n", data_normalized)
    concept = np.array([0.23, 0.06, 0.19, 0.25, 0.50, 0.19])
    weight_vector = np.array([0.2, 0.25, 0.30, 0.14, 0.03, 0.08])
    theta = 0.45
    rank_result = zhang_tfs_method(data_normalized, weight_vector, theta, concept)
    print("absolute排序结果:\n", rank_result[0])
    print("relative排序结果:\n", rank_result[1])
