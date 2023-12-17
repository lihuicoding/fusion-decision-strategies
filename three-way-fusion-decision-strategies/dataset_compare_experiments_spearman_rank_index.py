import numpy as np
import three_way_main_procedure
import spearman_and_kendall_rank_index
import inherent_rank_index
import local_ranking_index
import waa_method
import topsis_method
import promethee_method
import jia_and_liu_method
import zhang_tfs_method
import jiang_tfs_outranking_method

np.random.seed(3)
theta = 0.4
results = []
# ----------------------------------------------------------------------------------------------------------------------
# 1.computer_hardware
print("1.computer_hardware:")
file = r'dataset/computer_hardware.csv'
data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(2, 8)))
data_normalized = three_way_main_procedure.data_normalize(data, [0, 1, 1, 1, 1, 1])
n, m = data_normalized.shape
concepts = np.random.random((10, m))  # 随机生成10个准则模糊概念
weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
theta_vector = np.ones(concepts.shape[0]) * theta  # theta值
aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理，采用平均

# our_method
our_opt_method_classify_result, our_opt_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 0)
our_com_method_classify_result, our_com_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 1)
our_pes_method_classify_result, our_pes_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 2)

# waa_method
waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

# topsis_method
topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

# promethee
promethee_method_rank_result = promethee_method.promethee_method(data_normalized, weight_vector)

# jia_and_liu_method, 0-alpha, 1-beta
jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

# zhang_tfs_method, 0-absolute, 1-relative
zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                 aggregated_concept)

# jiang_tfs_outranking_method
jiang_tfs_outranking_method_rank_result = jiang_tfs_outranking_method.jiang_tfs_outranking_method(data_normalized,
                                                                                                  weight_vector, theta)

print("opt_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("com_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("pes_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

# ----------------------------------------------------------------------------------------------------------------------
# 2.tripadvisor_review
print("2.tripadvisor_review:")
file = r'dataset/tripadvisor_review.csv'
data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(1, 11)))
data_normalized = three_way_main_procedure.data_normalize(data, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# print("标准化后的数据:\n", data_normalized)
n, m = data_normalized.shape

concepts = np.random.random((10, m))
# print("concepts:\n", concepts)
weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
theta_vector = np.ones(concepts.shape[0]) * theta  # theta值
aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

# our_method
our_opt_method_classify_result, our_opt_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 0)
our_com_method_classify_result, our_com_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 1)
our_pes_method_classify_result, our_pes_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 2)

# waa_method
waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

# topsis_method
topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

# promethee
promethee_method_rank_result = promethee_method.promethee_method(data_normalized, weight_vector)

# jia_and_liu_method, 0-alpha, 1-beta
jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

# zhang_tfs_method, 0-absolute, 1-relative
zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                 aggregated_concept)

# jiang_tfs_outranking_method
jiang_tfs_outranking_method_rank_result = jiang_tfs_outranking_method.jiang_tfs_outranking_method(data_normalized,
                                                                                                  weight_vector, theta)

print("opt_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("com_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("pes_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

# 3.concrete_strength
print("3.concrete_strength:")
file = r'dataset/concrete_strength.csv'
data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(8)))
data_normalized = three_way_main_procedure.data_normalize(data, [1, 1, 1, 0, 1, 1, 1, 1])
# print("标准化后的数据:\n", data_normalized)
n, m = data_normalized.shape

concepts = np.random.random((10, m))
# print("concepts:\n", concepts)
weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
theta_vector = np.ones(concepts.shape[0]) * theta  # theta值
aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

# our_method
our_opt_method_classify_result, our_opt_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 0)
our_com_method_classify_result, our_com_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 1)
our_pes_method_classify_result, our_pes_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 2)

# waa_method
waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

# topsis_method
topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

# promethee
promethee_method_rank_result = promethee_method.promethee_method(data_normalized, weight_vector)

# jia_and_liu_method, 0-alpha, 1-beta
jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

# zhang_tfs_method, 0-absolute, 1-relative
zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                 aggregated_concept)

# jiang_tfs_outranking_method
jiang_tfs_outranking_method_rank_result = jiang_tfs_outranking_method.jiang_tfs_outranking_method(data_normalized,
                                                                                                  weight_vector, theta)

print("opt_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("com_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("pes_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

'''
# 4.winequality_red
print("4.winequality_red:")
file = r'dataset/winequality_red.csv'
# data = three_way_main_procedure.read_csv_data(file, 1, (1, 2, 7, 8, 9, 10))
# data_normalized = three_way_main_procedure.data_normalize(data, [0, 1, 0, 0, 1, 1])
data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(0, 11)))
data_normalized = three_way_main_procedure.data_normalize(data, [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1])
n, m = data_normalized.shape
concepts = np.random.random((10, m))
weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
theta_vector = np.ones(concepts.shape[0]) * theta  # theta值
aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

# our_method
our_opt_method_classify_result, our_opt_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 0)
our_com_method_classify_result, our_com_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 1)
our_pes_method_classify_result, our_pes_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 2)

# waa_method
waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

# topsis_method
topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

# promethee
promethee_method_rank_result = promethee_method.promethee_method(data_normalized, weight_vector)

# jia_and_liu_method, 0-alpha, 1-beta
jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

# zhang_tfs_method, 0-absolute, 1-relative
zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                 aggregated_concept)

# jiang_tfs_outranking_method
jiang_tfs_outranking_method_rank_result = jiang_tfs_outranking_method.jiang_tfs_outranking_method(data_normalized,
                                                                                                  weight_vector, theta)

print("opt_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, waa_method_rank_result)[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, topsis_method_rank_result)[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, promethee_method_rank_result)[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, jia_and_liu_method_rank_result[0])[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, jia_and_liu_method_rank_result[1])[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, zhang_tfs_method_rank_result[0])[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, zhang_tfs_method_rank_result[1])[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("com_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, waa_method_rank_result)[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, topsis_method_rank_result)[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, promethee_method_rank_result)[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, jia_and_liu_method_rank_result[0])[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, jia_and_liu_method_rank_result[1])[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, zhang_tfs_method_rank_result[0])[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, zhang_tfs_method_rank_result[1])[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("pes_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, waa_method_rank_result)[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, topsis_method_rank_result)[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, promethee_method_rank_result)[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, jia_and_liu_method_rank_result[0])[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, jia_and_liu_method_rank_result[1])[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, zhang_tfs_method_rank_result[0])[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, zhang_tfs_method_rank_result[1])[0],
       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")
'''

# 5.friedman
print("5.friedman:")
file = r'dataset/friedman.csv'
data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(0, 5)))
data_normalized = three_way_main_procedure.data_normalize(data, [1, 1, 1, 1, 1])
n, m = data_normalized.shape
concepts = np.random.random((10, m))
# print("concepts:\n", concepts)
weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
theta_vector = np.ones(concepts.shape[0]) * theta  # theta值
aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

# our_method
our_opt_method_classify_result, our_opt_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 0)
our_com_method_classify_result, our_com_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 1)
our_pes_method_classify_result, our_pes_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 2)

# waa_method
waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

# topsis_method
topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

# promethee
promethee_method_rank_result = promethee_method.promethee_method(data_normalized, weight_vector)

# jia_and_liu_method, 0-alpha, 1-beta
jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

# zhang_tfs_method, 0-absolute, 1-relative
zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                 aggregated_concept)

# jiang_tfs_outranking_method
jiang_tfs_outranking_method_rank_result = jiang_tfs_outranking_method.jiang_tfs_outranking_method(data_normalized,
                                                                                                  weight_vector, theta)

print("opt_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("com_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("pes_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

'''
# 6.treasury
print("6.treasury:")
file = r'dataset/treasury.csv'
data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(0, 15)))
data_normalized = three_way_main_procedure.data_normalize(data, [1 for i in range(0, 15)])
n, m = data_normalized.shape
concepts = np.random.random((10, m))
# print("concepts:\n", concepts)
weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
theta_vector = np.ones(concepts.shape[0]) * theta  # theta值
aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

# our_method
our_opt_method_classify_result, our_opt_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 0)
our_com_method_classify_result, our_com_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 1)
our_pes_method_classify_result, our_pes_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 2)

# waa_method
waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

# topsis_method
topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

# promethee
promethee_method_rank_result = promethee_method.promethee_method(data_normalized, weight_vector)

# jia_and_liu_method, 0-alpha, 1-beta
jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

# zhang_tfs_method, 0-absolute, 1-relative
zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                 aggregated_concept)

# jiang_tfs_outranking_method
jiang_tfs_outranking_method_rank_result = jiang_tfs_outranking_method.jiang_tfs_outranking_method(data_normalized,
                                                                                                  weight_vector, theta)

print("opt_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("com_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("pes_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")
'''

# 7.wizmir
print("7.wizmir:")
file = r'dataset/wizmir.csv'
data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(0, 9)))
data_normalized = three_way_main_procedure.data_normalize(data, [1, 1, 1, 1, 1, 1, 1, 1, 1])
n, m = data_normalized.shape
concepts = np.random.random((10, m))
# print("concepts:\n", concepts)
weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
theta_vector = np.ones(concepts.shape[0]) * theta  # theta值
aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

# our_method
our_opt_method_classify_result, our_opt_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 0)
our_com_method_classify_result, our_com_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 1)
our_pes_method_classify_result, our_pes_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 2)

# waa_method
waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

# topsis_method
topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

# promethee
promethee_method_rank_result = promethee_method.promethee_method(data_normalized, weight_vector)

# jia_and_liu_method, 0-alpha, 1-beta
jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

# zhang_tfs_method, 0-absolute, 1-relative
zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                 aggregated_concept)

# jiang_tfs_outranking_method
jiang_tfs_outranking_method_rank_result = jiang_tfs_outranking_method.jiang_tfs_outranking_method(data_normalized,
                                                                                                  weight_vector, theta)

print("opt_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("com_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("pes_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

'''
# 8.mortgage
print("8.mortgage:")
file = r'dataset/mortgage.csv'
data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(0, 15)))
data_normalized = three_way_main_procedure.data_normalize(data, [1 for i in range(0, 15)])
n, m = data_normalized.shape
concepts = np.random.random((10, m))
# print("concepts:\n", concepts)
weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
theta_vector = np.ones(concepts.shape[0]) * theta  # theta值
aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

# our_method
our_opt_method_classify_result, our_opt_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 0)
our_com_method_classify_result, our_com_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 1)
our_pes_method_classify_result, our_pes_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 2)

# waa_method
waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

# topsis_method
topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

# promethee
promethee_method_rank_result = promethee_method.promethee_method(data_normalized, weight_vector)

# jia_and_liu_method, 0-alpha, 1-beta
jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

# zhang_tfs_method, 0-absolute, 1-relative
zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                 aggregated_concept)

# jiang_tfs_outranking_method
jiang_tfs_outranking_method_rank_result = jiang_tfs_outranking_method.jiang_tfs_outranking_method(data_normalized,
                                                                                                  weight_vector, theta)

print("opt_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("com_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("pes_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")
'''

# 9.dee
print("9.dee:")
file = r'dataset/dee.csv'
data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(0, 6)))
data_normalized = three_way_main_procedure.data_normalize(data, [1, 1, 1, 1, 1, 1])
n, m = data_normalized.shape
concepts = np.random.random((10, m))
# print("concepts:\n", concepts)
weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
theta_vector = np.ones(concepts.shape[0]) * theta  # theta值
aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

# our_method
our_opt_method_classify_result, our_opt_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 0)
our_com_method_classify_result, our_com_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 1)
our_pes_method_classify_result, our_pes_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 2)

# waa_method
waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

# topsis_method
topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

# promethee
promethee_method_rank_result = promethee_method.promethee_method(data_normalized, weight_vector)

# jia_and_liu_method, 0-alpha, 1-beta
jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

# zhang_tfs_method, 0-absolute, 1-relative
zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                 aggregated_concept)

# jiang_tfs_outranking_method
jiang_tfs_outranking_method_rank_result = jiang_tfs_outranking_method.jiang_tfs_outranking_method(data_normalized,
                                                                                                  weight_vector, theta)

print("opt_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("com_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("pes_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

# 10.abalone
print("10.abalone:")
file = r'dataset/abalone.csv'
data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(1, 8)))
data_normalized = three_way_main_procedure.data_normalize(data, [1, 1, 1, 1, 1, 1, 1])
n, m = data_normalized.shape
concepts = np.random.random((10, m))
# print("concepts:\n", concepts)
weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
theta_vector = np.ones(concepts.shape[0]) * theta  # theta值
aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

# our_method
our_opt_method_classify_result, our_opt_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 0)
our_com_method_classify_result, our_com_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 1)
our_pes_method_classify_result, our_pes_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 2)

# waa_method
waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

# topsis_method
topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

# promethee
promethee_method_rank_result = promethee_method.promethee_method(data_normalized, weight_vector)

# jia_and_liu_method, 0-alpha, 1-beta
jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

# zhang_tfs_method, 0-absolute, 1-relative
zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                 aggregated_concept)

# jiang_tfs_outranking_method
jiang_tfs_outranking_method_rank_result = jiang_tfs_outranking_method.jiang_tfs_outranking_method(data_normalized,
                                                                                                  weight_vector, theta)

print("opt_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("com_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("pes_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

'''
# 11.winequality_white
print("11.winequality_white:")
file = r'dataset/winequality_white.csv'
data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(0, 11)))
data_normalized = three_way_main_procedure.data_normalize(data, [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1])
n, m = data_normalized.shape
concepts = np.random.random((10, m))
# print("concepts:\n", concepts)
weight_vector = np.array([1 / m for _ in range(m)])  # 属性权重
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
theta_vector = np.ones(concepts.shape[0]) * theta  # theta值
aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

# our_method
our_opt_method_classify_result, our_opt_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 0)
our_com_method_classify_result, our_com_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 1)
our_pes_method_classify_result, our_pes_method_rank_result = three_way_main_procedure. \
    three_way_model(data_normalized, concepts, theta_vector, weight_vector, expecter_weight_vector, 2)

# waa_method
waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

# topsis_method
topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

# promethee
promethee_method_rank_result = promethee_method.promethee_method(data_normalized, weight_vector)

# jia_and_liu_method, 0-alpha, 1-beta
jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

# zhang_tfs_method, 0-absolute, 1-relative
zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                 aggregated_concept)

# jiang_tfs_outranking_method
jiang_tfs_outranking_method_rank_result = jiang_tfs_outranking_method.jiang_tfs_outranking_method(data_normalized,
                                                                                                  weight_vector, theta)

print("opt_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_opt_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("com_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_com_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")

print("pes_method")
print("---------------------------------------------------------------------------------------------------------------")
row = [
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result, waa_method_rank_result)[
        0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    topsis_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    promethee_method_rank_result)[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jia_and_liu_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[0])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    zhang_tfs_method_rank_result[1])[0],
    spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_pes_method_rank_result,
                                                                    jiang_tfs_outranking_method_rank_result)[0]]
results.append(row)
print(row)
print("---------------------------------------------------------------------------------------------------------------")
'''

results = np.array(results)
np.savetxt("results\\dataset_compare_experiments_with_our_methods_spearman_rank_index_result.csv", results,
           delimiter=",",
           fmt='%.4f')
