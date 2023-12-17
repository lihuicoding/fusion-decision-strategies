import numpy as np
import waa_method
import topsis_method
import three_way_main_procedure
import spearman_and_kendall_rank_index
import jia_and_liu_method
import zhang_tfs_method

# ----------------------------------------------------------------------------------------------------------------------
# example
print("example:")
file = r'dataset/example.csv'
data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(6)))
data_normalized = three_way_main_procedure.data_normalize(data, [0, 1, 1, 1, 1, 1])
# print("标准化后的数据:\n", data_normalized)
n, m = data_normalized.shape

concepts = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                         [0.2, 0.6, 0.4, 0.3, 0.6, 0.4],
                         [0.3, 0.2, 0.7, 0.9, 0.1, 0.6],
                         [0.7,0.7, 0.5, 0.6, 0.4, 0.7]])
# concepts = np.random.random((16, 6))
# print("concepts:\n", concepts)
weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
theta_vector = np.ones(concepts.shape[0]) * 0.45
theta = 0.45
aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

# our_method
our_method_classify_result, our_method_rank_result = three_way_main_procedure.three_way_model(data_normalized, concepts,
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

# print("our_method_classify:", our_method_classify_result)
print("our_method_vs_waa:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result, waa_method_rank_result))
print("our_method_vs_topsis", spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                                              topsis_method_rank_result))
print("our_method_vs_jia_and_liu:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      jia_and_liu_method_rank_result[0]))
print("our_method_vs_zhang_tfs_absolute:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      zhang_tfs_method_rank_result[0]))
print("our_method_vs_zhang_tfs_relative:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      zhang_tfs_method_rank_result[1]))
print("---------------------------------------------------------------------------------------------------------------")

# ----------------------------------------------------------------------------------------------------------------------
# computer_hardware
print("computer_hardware:")
file = r'dataset/computer_hardware.csv'
data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(2, 8)))
data_normalized = three_way_main_procedure.data_normalize(data, [0, 1, 1, 1, 1, 1])
# print("标准化后的数据:\n", data_normalized)
n, m = data_normalized.shape

# concepts = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#                      [0.2, 0.6, 0.4, 0.3, 0.6, 0.4],
#                      [0.3, 0.2, 0.7, 0.9, 0.1, 0.6]])
concepts = np.random.random((16, 6))
# print("concepts:\n", concepts)
weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
theta_vector = np.ones(concepts.shape[0]) * 0.45
theta = 0.45
aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

# our_method
our_method_classify_result, our_method_rank_result = three_way_main_procedure.three_way_model(data_normalized, concepts,
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

print("our_method_classify:", our_method_classify_result)
print("our_method_vs_waa:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result, waa_method_rank_result))
print("our_method_vs_topsis", spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                                              topsis_method_rank_result))
print("topsis_vs_waa:", spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                                        waa_method_rank_result))
print("our_method_vs_jia_and_liu:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      jia_and_liu_method_rank_result[0]))
print("our_method_vs_zhang_tfs_absolute:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      zhang_tfs_method_rank_result[0]))
print("our_method_vs_zhang_tfs_relative:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      zhang_tfs_method_rank_result[1]))
print("---------------------------------------------------------------------------------------------------------------")

# ----------------------------------------------------------------------------------------------------------------------
# tripadvisor_review
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
our_method_classify_result, our_method_rank_result = three_way_main_procedure.three_way_model(data_normalized, concepts,
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

# print("our_method_classify:", our_method_classify_result)
print("our_method_vs_waa:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result, waa_method_rank_result))
print("our_method_vs_topsis", spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                                              topsis_method_rank_result))
print("topsis_vs_waa:", spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                                        waa_method_rank_result))
print("our_method_vs_jia_and_liu:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      jia_and_liu_method_rank_result[0]))
print("our_method_vs_zhang_tfs_absolute:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      zhang_tfs_method_rank_result[0]))
print("our_method_vs_zhang_tfs_relative:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      zhang_tfs_method_rank_result[1]))
print("---------------------------------------------------------------------------------------------------------------")

# concrete_strength
print("concrete_strength:")
file = r'dataset/concrete_strength.csv'
data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(8)))
data_normalized = three_way_main_procedure.data_normalize(data, [1, 1, 0, 0, 1, 1, 1, 1])
# print("标准化后的数据:\n", data_normalized)
n, m = data_normalized.shape

concepts = np.random.random((16, 8))
# print("concepts:\n", concepts)
weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
theta_vector = np.ones(concepts.shape[0]) * 0.45
theta = 0.45
aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

# our_method
our_method_classify_result, our_method_rank_result = three_way_main_procedure.three_way_model(data_normalized, concepts,
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

# print("our_method_classify:", our_method_classify_result)
print("our_method_vs_waa:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result, waa_method_rank_result))
print("our_method_vs_topsis", spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                                              topsis_method_rank_result))
print("topsis_vs_waa:", spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                                        waa_method_rank_result))
print("our_method_vs_jia_and_liu:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      jia_and_liu_method_rank_result[0]))
print("our_method_vs_zhang_tfs_absolute:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      zhang_tfs_method_rank_result[0]))
print("our_method_vs_zhang_tfs_relative:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      zhang_tfs_method_rank_result[1]))
print("---------------------------------------------------------------------------------------------------------------")

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
our_method_classify_result, our_method_rank_result = three_way_main_procedure.three_way_model(data_normalized, concepts,
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

# print("our_method_classify:", our_method_classify_result)
print("our_method_vs_waa:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result, waa_method_rank_result))
print("our_method_vs_topsis", spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                                              topsis_method_rank_result))
print("topsis_vs_waa:", spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                                        waa_method_rank_result))
print("our_method_vs_jia_and_liu:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      jia_and_liu_method_rank_result[0]))
print("our_method_vs_zhang_tfs_absolute:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      zhang_tfs_method_rank_result[0]))
print("our_method_vs_zhang_tfs_relative:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      zhang_tfs_method_rank_result[1]))
print("---------------------------------------------------------------------------------------------------------------")

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
our_method_classify_result, our_method_rank_result = three_way_main_procedure.three_way_model(data_normalized, concepts,
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

# print("our_method_classify:", our_method_classify_result)
print("our_method_vs_waa:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result, waa_method_rank_result))
print("our_method_vs_topsis", spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                                              topsis_method_rank_result))
print("topsis_vs_waa:", spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                                        waa_method_rank_result))
print("our_method_vs_jia_and_liu:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      jia_and_liu_method_rank_result[0]))
print("our_method_vs_zhang_tfs_absolute:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      zhang_tfs_method_rank_result[0]))
print("our_method_vs_zhang_tfs_relative:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_rank_result,
                                                                      zhang_tfs_method_rank_result[1]))
print("---------------------------------------------------------------------------------------------------------------")
