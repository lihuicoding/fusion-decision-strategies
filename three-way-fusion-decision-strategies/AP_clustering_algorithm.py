import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
from three_way_main_procedure import data_normalize, read_csv_data

if __name__ == '__main__':
    # 输入数据
    # file = r'dataset/computer_hardware.csv'
    # data = read_csv_data(file, 1, (i for i in range(2, 8)))
    # data_normalized = data_normalize(data, [0, 1, 1, 1, 1, 1])
    # print("标准化后的数据:\n", X)
    # print("tripadvisor_review:")
    # file = r'dataset/tripadvisor_review.csv'
    # data = read_csv_data(file, 1, (i for i in range(1, 11)))
    # data_normalized = data_normalize(data, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # print("winequality_red:")
    # file = r'dataset/winequality_red.csv'
    # data = read_csv_data(file, 1, (1, 2, 7, 8, 9, 10))
    # data_normalized = data_normalize(data, [0, 1, 0, 0, 1, 1])
    print("winequality_white:")
    file = r'dataset/winequality_white.csv'
    data = read_csv_data(file, 1, (7, 8, 4, 10))
    data_normalized = data_normalize(data, [0, 1, 0, 1])
    af = AffinityPropagation(preference=-0.5, random_state=0).fit(data_normalized)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    # 打印聚类结果
    print("聚类数目：%d" % n_clusters_)
    print("聚类中心：", cluster_centers_indices)
    print("聚类标签：", labels)

    # 计算每个对象的类
    neighbour_classes = []
    for i in range(data.shape[0]):
        neighbour_classes.append(np.where(labels == labels[i]))
    # print(neighbour_classes)