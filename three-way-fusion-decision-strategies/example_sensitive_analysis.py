""""
用整个数据集来进行敏感度分析
"""
import numpy as np
from three_way_main_procedure import *
from topsis_method import topsis
from spearman_and_kendall_rank_index import spearman_and_kendall_rank_index
from local_ranking_index import local_ranking_index
from inherent_rank_index import inherent_rank_index

# file = r'dataset/example.csv'
# data = read_csv_data(file, 1, (i for i in range(6)))
file = r'dataset/computer_hardware.csv'
data = read_csv_data(file, 1, (i for i in range(2, 8)))
data_normalized = data_normalize(data, [0, 1, 1, 1, 1, 1])
# print("标准化后的数据:\n", np.around(data_normalized, 4))

concepts = np.array([[0.5, 0.3, 0.5, 0.4, 0.7, 0.8],
                     [0.2, 0.6, 0.4, 0.7, 0.6, 0.4],
                     [0.3, 0.2, 0.7, 0.9, 0.1, 0.6],
                     [0.7, 0.7, 0.5, 0.6, 0.4, 0.7]])
attribute_weight_vector = np.array([0.15, 0.2, 0.1, 0.25, 0.1, 0.2])
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])
theta_vector = np.ones(concepts.shape[0]) * 0.36

topsis_rank_result = topsis(data_normalized, attribute_weight_vector)
result = []

for i in np.arange(0.01, 0.5, 0.01):
    theta_vector = np.ones(concepts.shape[0]) * i
    classify_result, opt_rank_result = three_way_model(data_normalized, concepts, theta_vector, attribute_weight_vector,
                                                       expecter_weight_vector, option=0)
    classify_result, com_rank_result = three_way_model(data_normalized, concepts, theta_vector, attribute_weight_vector,
                                                       expecter_weight_vector, option=1)
    classify_result, pes_rank_result = three_way_model(data_normalized, concepts, theta_vector, attribute_weight_vector,
                                                       expecter_weight_vector, option=2)
    # theta,opt_srcc, com_srcc, pes_srcc, opt_lri, com_lri, pes_lri, opt_rri, com_rri, pes_rri
    row = np.array([i,
                    spearman_and_kendall_rank_index(topsis_rank_result, opt_rank_result)[0],
                    spearman_and_kendall_rank_index(topsis_rank_result, com_rank_result)[0],
                    spearman_and_kendall_rank_index(topsis_rank_result, pes_rank_result)[0],
                    np.around(local_ranking_index(topsis_rank_result, opt_rank_result), 4),
                    np.around(local_ranking_index(topsis_rank_result, com_rank_result), 4),
                    np.around(local_ranking_index(topsis_rank_result, pes_rank_result), 4),
                    np.around(inherent_rank_index(data_normalized, opt_rank_result), 4),
                    np.around(inherent_rank_index(data_normalized, com_rank_result), 4),
                    np.around(inherent_rank_index(data_normalized, pes_rank_result), 4),
                    ])
    result.append(row)

result = np.array(result)
np.savetxt("results\\example_sensitive_analysis_result.csv", result, delimiter=",", fmt='%.4f')
