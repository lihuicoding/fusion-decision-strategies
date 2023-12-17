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

# ----------------------------------------------------------------------------------------------------------------------
# example
print("example:")
file = r'dataset/example.csv'
data = three_way_main_procedure.read_csv_data(file, 1, (i for i in range(6)))
data_normalized = three_way_main_procedure.data_normalize(data, [0, 1, 1, 1, 1, 1])
# print("标准化后的数据:\n", data_normalized)
n, m = data_normalized.shape

concepts = np.array([[0.5, 0.3, 0.5, 0.4, 0.7, 0.8],
                     [0.2, 0.6, 0.4, 0.7, 0.6, 0.4],
                     [0.3, 0.2, 0.7, 0.9, 0.1, 0.6],
                     [0.7, 0.7, 0.5, 0.6, 0.4, 0.7]])
# concepts = np.random.random((16, 6))
# print("concepts:\n", concepts)
weight_vector = np.array([0.15, 0.2, 0.1, 0.25, 0.1, 0.2])  # 属性权重
expecter_weight_vector = np.ones(concepts.shape[0]) * (1 / concepts.shape[0])  # 专家权重
theta_vector = np.ones(concepts.shape[0]) * 0.36
theta = 0.36
aggregated_concept = data_normalized.mean(axis=0)  # zhang_tfs处理

# our_method
our_method_opt_classify_result, our_method_opt_rank_result = three_way_main_procedure.three_way_model(data_normalized,
                                                                                                      concepts,
                                                                                                      theta_vector,
                                                                                                      weight_vector,
                                                                                                      expecter_weight_vector,
                                                                                                      option=0)
our_method_com_classify_result, our_method_com_rank_result = three_way_main_procedure.three_way_model(data_normalized,
                                                                                                      concepts,
                                                                                                      theta_vector,
                                                                                                      weight_vector,
                                                                                                      expecter_weight_vector,
                                                                                                      option=1)
our_method_pes_classify_result, our_method_pes_rank_result = three_way_main_procedure.three_way_model(data_normalized,
                                                                                                      concepts,
                                                                                                      theta_vector,
                                                                                                      weight_vector,
                                                                                                      expecter_weight_vector,
                                                                                                      option=2)
# waa_method
waa_method_rank_result = waa_method.waa(data_normalized, weight_vector)

# topsis_method
topsis_method_rank_result = topsis_method.topsis(data_normalized, weight_vector)

# jia_and_liu_method
jia_and_liu_method_rank_result = jia_and_liu_method.jia_and_liu_method(data_normalized, weight_vector, theta)

# zhang_tfs_method
zhang_tfs_method_rank_result = zhang_tfs_method.zhang_tfs_method(data_normalized, weight_vector, theta,
                                                                 aggregated_concept)

# jiang_tfs_outranking_method
jiang_tfs_outranking_method_rank_result = jiang_tfs_outranking_method.jiang_tfs_outranking_method(data_normalized,
                                                                                                  weight_vector, theta)

# promethee_method
promethee_method_rank_result = promethee_method.promethee_method(data_normalized, weight_vector)

print("排序结果")
print("---------------------------------------------------------------------------------------------------------------")
print("our_method_opt:", our_method_opt_rank_result)
print("our_method_com:", our_method_com_rank_result)
print("our_method_pes:", our_method_pes_rank_result)
print("waa_method:", waa_method_rank_result)
print("topsis_method:", topsis_method_rank_result)
print("promethee_method:", promethee_method_rank_result)
print("jia_and_liu_alpha:", jia_and_liu_method_rank_result[0])
print("jia_and_liu_beta:", jia_and_liu_method_rank_result[1])
print("zhang_tfs_absolute:", zhang_tfs_method_rank_result[0])
print("zhang_tfs_relative:", zhang_tfs_method_rank_result[1])
print("jiang_tfs_outranking:", jiang_tfs_outranking_method_rank_result)
print("---------------------------------------------------------------------------------------------------------------")

print("")
print("spearman index:")
print("opt model:")
print("---------------------------------------------------------------------------------------------------------------")
print("our_method_vs_waa:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_opt_rank_result,
                                                                      waa_method_rank_result))
print("our_method_vs_topsis",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_opt_rank_result,
                                                                      topsis_method_rank_result))
print("our_method_vs_promethee:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_opt_rank_result,
                                                                      promethee_method_rank_result))
print("our_method_vs_jia_and_liu_alpha:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_opt_rank_result,
                                                                      jia_and_liu_method_rank_result[0]))
print("our_method_vs_jia_and_liu_beta:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_opt_rank_result,
                                                                      jia_and_liu_method_rank_result[1]))
print("our_method_vs_zhang_tfs_absolute:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_opt_rank_result,
                                                                      zhang_tfs_method_rank_result[0]))
print("our_method_vs_zhang_tfs_relative:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_opt_rank_result,
                                                                      zhang_tfs_method_rank_result[1]))
print("our_method_vs_jiang_tfs_outranking:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_opt_rank_result,
                                                                      jiang_tfs_outranking_method_rank_result))
print("---------------------------------------------------------------------------------------------------------------")

print("com model:")
print("---------------------------------------------------------------------------------------------------------------")
print("our_method_vs_waa:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_com_rank_result,
                                                                      waa_method_rank_result))
print("our_method_vs_topsis",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_com_rank_result,
                                                                      topsis_method_rank_result))
print("our_method_vs_promethee:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_com_rank_result,
                                                                      promethee_method_rank_result))
print("our_method_vs_jia_and_liu_alpha:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_com_rank_result,
                                                                      jia_and_liu_method_rank_result[0]))
print("our_method_vs_jia_and_liu_beta:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_com_rank_result,
                                                                      jia_and_liu_method_rank_result[1]))
print("our_method_vs_zhang_tfs_absolute:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_com_rank_result,
                                                                      zhang_tfs_method_rank_result[0]))
print("our_method_vs_zhang_tfs_relative:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_com_rank_result,
                                                                      zhang_tfs_method_rank_result[1]))
print("our_method_vs_jiang_tfs_outranking:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_com_rank_result,
                                                                      jiang_tfs_outranking_method_rank_result))
print("---------------------------------------------------------------------------------------------------------------")

print("pes model:")
print("---------------------------------------------------------------------------------------------------------------")
# print("our_method_classify:", our_method_classify_result)
print("our_method_vs_waa:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_pes_rank_result,
                                                                      waa_method_rank_result))
print("our_method_vs_topsis",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_pes_rank_result,
                                                                      topsis_method_rank_result))
print("our_method_vs_promethee:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_pes_rank_result,
                                                                      promethee_method_rank_result))
print("our_method_vs_jia_and_liu_alpha:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_pes_rank_result,
                                                                      jia_and_liu_method_rank_result[0]))
print("our_method_vs_jia_and_liu_beta:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_pes_rank_result,
                                                                      jia_and_liu_method_rank_result[1]))
print("our_method_vs_zhang_tfs_absolute:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_pes_rank_result,
                                                                      zhang_tfs_method_rank_result[0]))
print("our_method_vs_zhang_tfs_relative:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_pes_rank_result,
                                                                      zhang_tfs_method_rank_result[1]))
print("our_method_vs_jiang_tfs_outranking:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_pes_rank_result,
                                                                      jiang_tfs_outranking_method_rank_result))
print("---------------------------------------------------------------------------------------------------------------")

print("")
print("inherent rank index:")
print("---------------------------------------------------------------------------------------------------------------")
print("our_method_opt",
      inherent_rank_index.inherent_rank_index(data_normalized, our_method_opt_rank_result))
print("our_method_com",
      inherent_rank_index.inherent_rank_index(data_normalized, our_method_com_rank_result))
print("our_method_pes",
      inherent_rank_index.inherent_rank_index(data_normalized, our_method_pes_rank_result))
print("waa:",
      inherent_rank_index.inherent_rank_index(data_normalized, waa_method_rank_result))
print("topsis",
      inherent_rank_index.inherent_rank_index(data_normalized, topsis_method_rank_result))
print("promethee:",
      inherent_rank_index.inherent_rank_index(data_normalized, promethee_method_rank_result))
print("jia_and_liu_alpha:",
      inherent_rank_index.inherent_rank_index(data_normalized, jia_and_liu_method_rank_result[0]))
print("jia_and_liu_beta:",
      inherent_rank_index.inherent_rank_index(data_normalized, jia_and_liu_method_rank_result[1]))
print("zhang_tfs_absolute:",
      inherent_rank_index.inherent_rank_index(data_normalized, zhang_tfs_method_rank_result[0]))
print("zhang_tfs_relative:",
      inherent_rank_index.inherent_rank_index(data_normalized, zhang_tfs_method_rank_result[1]))
print("jiang_tfs_outranking:",
      inherent_rank_index.inherent_rank_index(data_normalized, jiang_tfs_outranking_method_rank_result))
print("---------------------------------------------------------------------------------------------------------------")

print("local ranking index:")
print("opt model:")
print("---------------------------------------------------------------------------------------------------------------")
# print("our_method_classify:", our_method_classify_result)
print("our_method_vs_waa:",
      local_ranking_index.local_ranking_index(our_method_opt_rank_result, waa_method_rank_result))
print("our_method_vs_topsis",
      local_ranking_index.local_ranking_index(our_method_opt_rank_result, topsis_method_rank_result))
print("our_method_vs_promethee:",
      local_ranking_index.local_ranking_index(our_method_opt_rank_result, promethee_method_rank_result))
print("our_method_vs_jia_and_liu_alpha:",
      local_ranking_index.local_ranking_index(our_method_opt_rank_result, jia_and_liu_method_rank_result[0]))
print("our_method_vs_jia_and_liu_beta:",
      local_ranking_index.local_ranking_index(our_method_opt_rank_result, jia_and_liu_method_rank_result[1]))
print("our_method_vs_zhang_tfs_absolute:",
      local_ranking_index.local_ranking_index(our_method_opt_rank_result, zhang_tfs_method_rank_result[0]))
print("our_method_vs_zhang_tfs_relative:",
      local_ranking_index.local_ranking_index(our_method_opt_rank_result, zhang_tfs_method_rank_result[1]))
print("our_method_vs_jiang_tfs_outranking:",
      local_ranking_index.local_ranking_index(our_method_opt_rank_result, jiang_tfs_outranking_method_rank_result))
print("---------------------------------------------------------------------------------------------------------------")

print("com model:")
print("---------------------------------------------------------------------------------------------------------------")
# print("our_method_classify:", our_method_classify_result)
print("our_method_vs_waa:",
      local_ranking_index.local_ranking_index(our_method_com_rank_result, waa_method_rank_result))
print("our_method_vs_topsis",
      local_ranking_index.local_ranking_index(our_method_com_rank_result, topsis_method_rank_result))
print("our_method_vs_promethee:",
      local_ranking_index.local_ranking_index(our_method_com_rank_result, promethee_method_rank_result))
print("our_method_vs_jia_and_liu_alpha:",
      local_ranking_index.local_ranking_index(our_method_com_rank_result, jia_and_liu_method_rank_result[0]))
print("our_method_vs_jia_and_liu_beta:",
      local_ranking_index.local_ranking_index(our_method_com_rank_result, jia_and_liu_method_rank_result[1]))
print("our_method_vs_zhang_tfs_absolute:",
      local_ranking_index.local_ranking_index(our_method_com_rank_result, zhang_tfs_method_rank_result[0]))
print("our_method_vs_zhang_tfs_relative:",
      local_ranking_index.local_ranking_index(our_method_com_rank_result, zhang_tfs_method_rank_result[1]))
print("our_method_vs_jiang_tfs_outranking:",
      local_ranking_index.local_ranking_index(our_method_com_rank_result, jiang_tfs_outranking_method_rank_result))
print("---------------------------------------------------------------------------------------------------------------")

print("pes model:")
print("---------------------------------------------------------------------------------------------------------------")
# print("our_method_classify:", our_method_classify_result)
print("our_method_vs_waa:",
      local_ranking_index.local_ranking_index(our_method_pes_rank_result, waa_method_rank_result))
print("our_method_vs_topsis",
      local_ranking_index.local_ranking_index(our_method_pes_rank_result, topsis_method_rank_result))
print("our_method_vs_promethee:",
      local_ranking_index.local_ranking_index(our_method_pes_rank_result, promethee_method_rank_result))
print("our_method_vs_jia_and_liu_alpha:",
      local_ranking_index.local_ranking_index(our_method_pes_rank_result, jia_and_liu_method_rank_result[0]))
print("our_method_vs_jia_and_liu_beta:",
      local_ranking_index.local_ranking_index(our_method_pes_rank_result, jia_and_liu_method_rank_result[1]))
print("our_method_vs_zhang_tfs_absolute:",
      local_ranking_index.local_ranking_index(our_method_pes_rank_result, zhang_tfs_method_rank_result[0]))
print("our_method_vs_zhang_tfs_relative:",
      local_ranking_index.local_ranking_index(our_method_pes_rank_result, zhang_tfs_method_rank_result[1]))
print("our_method_vs_jiang_tfs_outranking:",
      local_ranking_index.local_ranking_index(our_method_pes_rank_result, jiang_tfs_outranking_method_rank_result))
print("---------------------------------------------------------------------------------------------------------------")

# print("---------------------------------------------------------------------------------------------------------------")
# print("与实际排序结果之间的比较：")
# print("SRCC:")
# real_rank = np.array([10, 4, 8, 6, 12, 11, 9, 3, 7, 2, 5, 1])
# print("our_opt_method",
#       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(real_rank, our_method_opt_rank_result))
# print("our_com_method",
#       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(real_rank, our_method_com_rank_result))
# print("our_pes_method",
#       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(real_rank, our_method_pes_rank_result))
# print("topsis",
#       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(our_method_opt_rank_result,
#                                                                       topsis_method_rank_result))
# print("promethee:",
#       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(real_rank,
#                                                                       promethee_method_rank_result))
# print("jia_and_liu_alpha:",
#       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(real_rank,
#                                                                       jia_and_liu_method_rank_result[0]))
# print("jia_and_liu_beta:",
#       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(real_rank,
#                                                                       jia_and_liu_method_rank_result[1]))
# print("zhang_tfs_absolute:",
#       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(real_rank,
#                                                                       zhang_tfs_method_rank_result[0]))
# print("zhang_tfs_relative:",
#       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(real_rank,
#                                                                       zhang_tfs_method_rank_result[1]))
# print("jiang_tfs_outranking:",
#       spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(real_rank,
#                                                                       jiang_tfs_outranking_method_rank_result))
# print("---------------------------------------------------------------------------------------------------------------")

print("---------------------------------------------------------------------------------------------------------------")
print("与topsis排序结果之间的比较：")
print("SRCC:")
print("---------------------------------------------------------------------------------------------------------------")
print("our_opt_method",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                      our_method_opt_rank_result))
print("our_com_method",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                      our_method_com_rank_result))
print("our_pes_method",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                      our_method_pes_rank_result))
print("waa:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                      waa_method_rank_result))
print("topsis",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                      topsis_method_rank_result))
print("promethee:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                      promethee_method_rank_result))
print("jia_and_liu_alpha:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                      jia_and_liu_method_rank_result[0]))
print("jia_and_liu_beta:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                      jia_and_liu_method_rank_result[1]))
print("zhang_tfs_absolute:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                      zhang_tfs_method_rank_result[0]))
print("zhang_tfs_relative:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                      zhang_tfs_method_rank_result[1]))
print("jiang_tfs_outranking:",
      spearman_and_kendall_rank_index.spearman_and_kendall_rank_index(topsis_method_rank_result,
                                                                      jiang_tfs_outranking_method_rank_result))
print("---------------------------------------------------------------------------------------------------------------")
print("local rank index:")
print("---------------------------------------------------------------------------------------------------------------")
print("our_opt_method",
      local_ranking_index.local_ranking_index(topsis_method_rank_result, our_method_opt_rank_result))
print("our_com_method",
      local_ranking_index.local_ranking_index(topsis_method_rank_result, our_method_com_rank_result))
print("our_pes_method",
      local_ranking_index.local_ranking_index(topsis_method_rank_result, our_method_pes_rank_result))
print("waa:",
      local_ranking_index.local_ranking_index(topsis_method_rank_result, waa_method_rank_result))
print("topsis",
      local_ranking_index.local_ranking_index(topsis_method_rank_result,
                                              topsis_method_rank_result))
print("promethee:",
      local_ranking_index.local_ranking_index(topsis_method_rank_result,
                                              promethee_method_rank_result))
print("jia_and_liu_alpha:",
      local_ranking_index.local_ranking_index(topsis_method_rank_result,
                                              jia_and_liu_method_rank_result[0]))
print("jia_and_liu_beta:",
      local_ranking_index.local_ranking_index(topsis_method_rank_result,
                                              jia_and_liu_method_rank_result[1]))
print("zhang_tfs_absolute:",
      local_ranking_index.local_ranking_index(topsis_method_rank_result,
                                              zhang_tfs_method_rank_result[0]))
print("zhang_tfs_relative:",
      local_ranking_index.local_ranking_index(topsis_method_rank_result,
                                              zhang_tfs_method_rank_result[1]))
print("jiang_tfs_outranking:",
      local_ranking_index.local_ranking_index(topsis_method_rank_result,
                                              jiang_tfs_outranking_method_rank_result))
print("---------------------------------------------------------------------------------------------------------------")
