import numpy as np
from sklearn.cluster import AffinityPropagation

'''
多准则模糊概念下的三支决策模型
'''


def read_csv_data(file_path, row_index, col_index):
    """
    读取csv文件中的数据,转换成一个numpy二维数组
    :param file_path: 文件路径
    :param row_index: 从哪一行开始
    :param col_index: 列索引，需要取哪些列，元组类型(0, 1, 3, 5),None代表全部
    :return: numpy_array
    """
    with open(file_path, encoding='utf-8') as f:
        data = np.loadtxt(file_path, str, delimiter=',', skiprows=row_index, usecols=col_index)
        data = data.astype(np.float64)
        return data


def write_csv_data(file_dir, file_name, data):
    """
    :param file_dir:
    :param file_name:
    :param data:
    :return:
    """
    pass


def data_normalize(original_data, is_benefit):
    """
    数据标准化，正规化
    :param is_benefit:是否是效用属性[1,0, 1, 0...],传入一个列表
    :param original_data: 原始数据
    :return:data_normalized
    """
    n, m = original_data.shape
    is_benefit = np.array(is_benefit)
    data_normalized = np.zeros_like(original_data)
    data_max = original_data.max(axis=0)
    data_min = original_data.min(axis=0)
    for i in range(n):
        data_normalized[i, is_benefit == 1] = (original_data[i, is_benefit == 1] - data_min[is_benefit == 1]) / (
                data_max[is_benefit == 1] - data_min[is_benefit == 1])  # 效用属性
        data_normalized[i, is_benefit == 0] = (data_max[is_benefit == 0] - original_data[i, is_benefit == 0]) / (
                data_max[is_benefit == 0] - data_min[is_benefit == 0])  # 成本属性
    return data_normalized


def multi_concepts_aggregated(concepts):
    """
    聚合准则模糊概念，分三种方式，乐观，悲观，折衷
    :param concepts: 准则模糊概念矩阵nxm
    :return: 聚合后，三个准则模糊概念构成的矩阵
    """
    n, m = concepts.shape
    neg = concepts.max(axis=0)  # 按列取最大的
    pos = concepts.min(axis=0)  # 按列取最小的
    mid = concepts.mean(axis=0)  # 取平均值
    return [pos, mid, neg]


def loss_functions(data, concept, theta, weight_vector):
    """
    计算损失函数
    :param data: 多属性决策表nxm
    :param concept: 准则模糊概念
    :param theta: 用来计算Action_B的损失参数，即损失规避因子
    :param weight_vector: 属性权重
    :return: n个对象的损失函数
    """
    n, m = data.shape
    loss_functions = []
    for i in range(n):
        PP = 0
        NP = np.sum(
            weight_vector * (1 / (1 + np.exp((data[i] - concept) * (-5)))))
        BP = theta * NP
        PN = np.sum(
            weight_vector * (1 / (1 + np.exp((concept - data[i]) * (-5)))))
        BN = theta * PN
        NN = 0
        loss_functions.append([[PP, PN], [BP, BN], [NP, NN]])
        # print(NP+PN)
    return np.array(loss_functions)


def multi_loss_functions_aggregated(data, concepts, theta_vector, attribute_weight_vector, experter_weight_vector):
    """
    多个损失函数聚合
    :param data: 原始数据
    :param concepts: 不同专家设置的准则模糊概念
    :param theta_vector: 不同专家的风险偏好
    :param attribute_weight_vector: 属性权重向量，都一样的
    :param expert_weight_vector: 每个专家的权重，权重向量设置为 1/k 时，即为取平均值
    :return: 聚合后所有对象的损失函数
    """
    n, m = data.shape  # n代表对象的数量，m代表属性的数量
    k, m = concepts.shape  # n代表专家的个数，m表示属性的个数
    aggregated_loss_functions = np.zeros((n, 3, 2))
    for j in range(k):
        loss = loss_functions(data, concepts[j], theta_vector[j], attribute_weight_vector)  # 一个专家下所有对象的损失函数
        aggregated_loss_functions = aggregated_loss_functions + loss * experter_weight_vector[
            j]  # 当权重向量设置为 1/k 时，即为取平均值
    return aggregated_loss_functions


def condition_probability(data, concept, weight_vector):
    """
    计算每个对象的条件概率
    :param data: 多属性决策表nxm
    :param concept: 准则模糊概念
    :param weight_vector: 属性权重
    :return: 每个对象的条件概率
    """
    n, m = data.shape

    # 邻域类算法计算对象的类
    # 计算加权欧式距离矩阵,得到相似矩阵
    distance_matrix = np.zeros((n, n)) * 0.0
    for i in range(n):
        for j in range(n):
            distance = np.sum((data[i] - data[j]) ** 2 * weight_vector)
            distance_matrix[i, j] = distance
    distance_matrix = -distance_matrix
    # #  求邻域
    # neighbour_classes = []
    # for i in range(n):
    #     neighbour_classes.append(np.where(distance_matrix[i] <= neighbour_radius)[0])

    # AP聚类算法计算对象的类
    # af = AffinityPropagation(random_state=0).fit(data) # 使用模型自带的欧氏距离
    af = AffinityPropagation(affinity='precomputed', random_state=0, max_iter=10000).fit(distance_matrix)  # 自定义距离公式，须传入相似矩阵
    # cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    # n_clusters_ = len(cluster_centers_indices)
    neighbour_classes = []
    for i in range(n):
        neighbour_classes.append(np.where(labels == labels[i])[0])
    # print(neighbour_classes)
    # 计算条件概率
    probabilities = np.zeros(n, dtype=np.float64)
    for index, neighbour_class in enumerate(neighbour_classes):
        for obj in neighbour_class:
            probabilities[index] += np.sum(
                weight_vector * np.minimum(data[obj], concept) / concept)  # 对应位置取小
        probabilities[index] = probabilities[index] / neighbour_class.size
    return probabilities


def condition_probability_under_multi_concepts(data, concepts, weight_vector):
    """
    多概念下的条件概率计算方式
    :param data:
    :param concepts: 多个准则模糊概念
    :param weight_vector: 属性权重
    :param neighbour_radius: 领域半径
    :return: 三种偏好的条件概率pos,neg,mid
    """
    aggregated_concepts = multi_concepts_aggregated(concepts)
    pos_probability = condition_probability(data, aggregated_concepts[0], weight_vector)
    mid_probability = condition_probability(data, aggregated_concepts[1], weight_vector)
    neg_probability = condition_probability(data, aggregated_concepts[2], weight_vector)
    return [pos_probability, mid_probability, neg_probability]


def three_way_model(data, concepts, theta_vector, attribute_weight_vector, experter_weight_vector, option):
    """
    单准则模糊概念下的三支决策模型
    :param data: 多属性决策表nxm
    :param concept: 准则模糊概念
    :param weight_vector: 属性权重
    :param theta: 用来计算Action_B的损失参数，即损失规避因子
    :param option: 策略， 0-opt,1-com,2-pes
    :return: classify_result, rank_result
    """
    n, m = data.shape
    # 通过损失函数计算阈值
    loss = multi_loss_functions_aggregated(data, concepts, theta_vector, attribute_weight_vector,
                                           experter_weight_vector)
    alpha = (loss[:, 0, 1] - loss[:, 1, 1]) / (loss[:, 0, 1] - loss[:, 1, 1] + loss[:, 1, 0] - loss[:, 0, 0])
    beta = (loss[:, 1, 1] - loss[:, 2, 1]) / (loss[:, 1, 1] - loss[:, 2, 1] + loss[:, 2, 0] - loss[:, 1, 0])
    gamma = (loss[:, 0, 1] - loss[:, 2, 1]) / (loss[:, 0, 1] - loss[:, 2, 1] + loss[:, 2, 0] - loss[:, 0, 0])
    # 条件概率
    probability = condition_probability_under_multi_concepts(data, concepts, attribute_weight_vector)[
        option]
    # 对象分类,where返回的是一个元组，所以取索引0
    pos = np.where(probability >= alpha)[0]  # 正域
    bnd = np.where((beta < probability) & (probability < alpha))[0]  # 边界域
    neg = np.where(probability <= beta)[0]  # 负域
    # 计算期望损失
    expect_loss = np.zeros((n, 3))
    for i in range(n):
        expect_loss[i, 0] = np.sum(loss[i, 0] * np.array([probability[i], 1 - probability[i]]))
        expect_loss[i, 1] = np.sum(loss[i, 1] * np.array([probability[i], 1 - probability[i]]))
        expect_loss[i, 2] = np.sum(loss[i, 2] * np.array([probability[i], 1 - probability[i]]))
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
    classify_result = (pos + 1, bnd + 1, neg + 1) # 处理的时候都是从0开始，所以返回结果要加1,负域中结果相反

    return [classify_result, rank_result + 1]


'''
def multi_concepts_three_way_model(data, concepts, weight_vector, theta, neighbour_radius):
    """
    多概念三支决策模型
    :param data: 多属性决策表nxm
    :param concepts: 准则模糊概念集
    :param weight_vector: 属性权重
    :param theta: 用来计算Action_B的损失参数，即损失规避因子
    :param neighbour_radius: 构造邻域类的时候的邻域半径
    :return: 三种 classify_result, rank_result
    """
    concepts = np.array(concepts)
    aggregated_concepts = multi_concepts_aggregated(concepts)  # 返回三个聚合后的准则模糊概念列表
    pos_result = three_way_model(data, aggregated_concepts[0], weight_vector, theta, neighbour_radius)
    neg_result = three_way_model(data, aggregated_concepts[1], weight_vector, theta, neighbour_radius)
    mid_result = three_way_model(data, aggregated_concepts[2], weight_vector, theta, neighbour_radius)
    return [pos_result, neg_result, mid_result]
'''

if __name__ == '__main__':
    file = r'dataset/example.csv'
    data = read_csv_data(file, 1, (i for i in range(6)))
    data_normalized = data_normalize(data, [0, 1, 1, 1, 1, 1])
    print("标准化后的数据:\n", np.around(data_normalized, 4))
    # concepts = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    #                      [0.2, 0.6, 0.4, 0.3, 0.6, 0.4],
    #                      [0.3, 0.2, 0.7, 0.9, 0.1, 0.6]])
    # concepts = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    #                      [0.2, 0.6, 0.4, 0.3, 0.6, 0.4],
    #                      [0.3, 0.2, 0.7, 0.9, 0.1, 0.6],
    #                      [0.7,0.7, 0.5, 0.6, 0.4, 0.7]])
    concepts = np.array([[0.5, 0.3, 0.5, 0.4, 0.7, 0.8],
                         [0.2, 0.6, 0.4, 0.7, 0.6, 0.4],
                         [0.3, 0.2, 0.7, 0.9, 0.1, 0.6],
                         [0.7,0.7, 0.5, 0.6, 0.4, 0.7]])
    # aggregated_concept = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    # concept = np.array([0.35, 0.55, 0.50, 0.65, 0.6, 0.45])
    # attribute_weight_vector = np.array([0.2, 0.25, 0.30, 0.14, 0.03, 0.08])
    attribute_weight_vector = np.array([0.15, 0.2, 0.1, 0.25, 0.1, 0.2])
    expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])
    theta_vector = np.ones(concepts.shape[0]) * 0.36

    classify_result, rank_result = three_way_model(data_normalized, concepts, theta_vector, attribute_weight_vector,
                                                   expecter_weight_vector, option=0)
    print("opt分类结果:\n", classify_result)
    print("opt排序结果:\n", rank_result)

    classify_result, rank_result = three_way_model(data_normalized, concepts, theta_vector, attribute_weight_vector,
                                                   expecter_weight_vector, option=1)
    print("com分类结果:\n", classify_result)
    print("com排序结果:\n", rank_result)

    classify_result, rank_result = three_way_model(data_normalized, concepts, theta_vector, attribute_weight_vector,
                                                   expecter_weight_vector, option=2)
    print("pes分类结果:\n", classify_result)
    print("pes排序结果:\n", rank_result)

