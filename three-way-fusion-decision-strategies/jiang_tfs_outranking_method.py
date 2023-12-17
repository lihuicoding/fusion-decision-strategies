import numpy as np
import three_way_main_procedure


def outranking_relation(data, weight_vector):
    """
    JiangHaiBo基于超序关系的三支决策方法Three-way multi-attribute decision-making based on outranking relations
    :param data: 标准化之后的数据
    :param weight_vector: 属性权重
    :return: 超序关系
    """
    n, m = data.shape
    result = [[] for i in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                result[i].append(j)
                continue
            cp = np.where(data[i] > data[j])[0]  # c_ij^+
            ce = np.where(data[i] == data[j])[0]  # c_ij^=
            cn = np.where(data[i] < data[j])[0]  # c_ij^-
            c = np.sum(weight_vector[cp]) + np.sum(weight_vector[ce])  # c_ij
            d = 2  # d_ij默认大于1即可
            if np.sum(weight_vector[cn]) != 0:
                d = np.sum(weight_vector[cp]) / np.sum(weight_vector[cn])  # d_ij
            sigma = 0.6  # c的阈值
            # 非一致性检验指数g_ij<1
            p = 0.2  # 投票阈值
            g = np.max((data[i] - data[j]) / p)
            if c >= sigma and d >= 1 and g >= 1:  # 满足这三个条件才满足关系
                result[i].append(j)

    return result


def jiang_tfs_outranking_method(data, weight_vector, theta):
    """
    :param data: 标准化之后的数据
    :param weight_vector: 属性权重
    :param theta:
    :return: average方式的排序结果
    """
    # 1.损失函数的计算
    n, m = data.shape
    data_max = data.max(axis=0)
    data_min = data.min(axis=0)
    theta = np.array([theta for i in range(m)])
    alpha, beta, gamma = np.zeros(n), np.zeros(n), np.zeros(n)
    # 先计算每个对象的损失函数
    loss_function = np.zeros((n, 3, 2))
    for i in range(n):
        loss_function[i, 0, 0] = 0  # PP
        loss_function[i, 0, 1] = np.sum(weight_vector * (data_max - data[i]))  # PN
        loss_function[i, 1, 0] = np.sum(theta * weight_vector * (data[i] - data_min))  # BP
        loss_function[i, 1, 1] = np.sum(theta * weight_vector * (data_max - data[i]))  # BN
        loss_function[i, 2, 0] = np.sum(weight_vector * (data[i] - data_min))  # NP
        loss_function[i, 2, 1] = 0  # NN

    object_class = outranking_relation(data, weight_vector)  # 超序类
    for i in range(n):
        # 损失函数聚合取平均
        PN = np.mean(loss_function[object_class[i], 0, 1])
        BP = np.mean(loss_function[object_class[i], 1, 0])
        BN = np.mean(loss_function[object_class[i], 1, 1])
        NP = np.mean(loss_function[object_class[i], 2, 0])
        loss_function[i, 0, 1] = PN
        loss_function[i, 1, 0] = BP
        loss_function[i, 1, 1] = BN
        loss_function[i, 2, 0] = NP
        # 计算阈值
        alpha[i] = (PN - BN) / ((PN - BN) + (BP - 0))
        beta[i] = (BN - 0) / ((BN - 0) + (NP - BP))
    # print(alpha)
    # print(beta)
    # 模糊概念的计算,加权求和
    concept = np.zeros(n)
    for i in range(n):
        concept[i] = np.sum(data[i] * weight_vector)

    # 计算条件概率
    condition_probability = np.zeros(n)
    for i in range(n):
        condition_probability[i] = np.sum(concept[object_class[i]]) / len(object_class[i])

    # 三支分类
    pos = np.where(condition_probability >= alpha)[0]  # 正域
    bnd = np.where((beta < condition_probability) & (condition_probability < alpha))[0]  # 边界域
    neg = np.where(condition_probability <= beta)[0]  # 负域
    # print(pos, bnd, neg)
    # 计算期望损失
    expect_loss = np.zeros((n, 3))
    for i in range(n):
        expect_loss[i, 0] = np.sum(
            loss_function[i, 0] * np.array([condition_probability[i], 1 - condition_probability[i]]))
        expect_loss[i, 1] = np.sum(
            loss_function[i, 1] * np.array([condition_probability[i], 1 - condition_probability[i]]))
        expect_loss[i, 2] = np.sum(
            loss_function[i, 2] * np.array([condition_probability[i], 1 - condition_probability[i]]))
    final_expect_loss = np.zeros(n)
    final_expect_loss[pos] = expect_loss[pos, 0]
    final_expect_loss[bnd] = expect_loss[bnd, 1]
    final_expect_loss[neg] = expect_loss[neg, 2]
    # 排序
    pos_rank = np.argsort(final_expect_loss[pos])  # 正域中的排序
    bnd_rank = np.argsort(final_expect_loss[bnd])  # 边界域中的排序
    neg_rank = np.argsort(final_expect_loss[neg])  # 负域中的排序
    rank_result = np.concatenate((pos[pos_rank], bnd[bnd_rank],
                                  neg[neg_rank][::-1]))  # 负域中的对象的排序规则与正域中的排序规则相反

    return rank_result + 1


if __name__ == "__main__":
    file = r'dataset/example.csv'
    data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(6)))
    # data = np.array([[96, 93, 90, 71, 85, 72],
    #                  [92, 82, 78, 72, 94, 78],
    #                  [85, 97, 84, 89, 92, 80],
    #                  [90, 83, 89, 71, 78, 85],
    #                  [98, 85, 82, 85, 69, 75],
    #                  [85, 89, 78, 68, 84, 88],
    #                  [84, 92, 81, 80, 72, 77],
    #                  [82, 85, 70, 75, 93, 67]], dtype=np.float64)
    data_normalized = three_way_main_procedure.data_normalize(data, [0, 1, 1, 1, 1, 1])
    # print("标准化后的数据:\n", data_normalized)
    weight_vector = np.array([0.15, 0.2, 0.1, 0.25, 0.1, 0.2])
    theta = 0.36
    # print(outranking_relation(data_normalized, weight_vector))
    print(jiang_tfs_outranking_method(data_normalized, weight_vector, theta))
