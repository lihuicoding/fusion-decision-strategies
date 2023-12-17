import numpy as np
from scipy.stats import friedmanchisquare
# import Orange  # version: Orange3==3.32.0
from Orange.evaluation.scoring import compute_CD
from Orange.evaluation.scoring import graph_ranks
import matplotlib.pyplot as plt

# 创建一个示例的性能指标矩阵，每行代表一个数据集，每列代表一个方法
# 10个方法，7个数据集的示例数据
performance_matrix = np.array([
    [0.9612, 0.9184, 0.9614, 0.9614, 0.8208, 0.8698, 0.8527, 0.9201, 0.9443, 0.9459],
    [0.9499, 0.8504, 0.9500, 0.9500, 0.6713, 0.7404, 0.8466, 0.7388, 0.7567, 0.9056],
    [0.9329, 0.8644, 0.9329, 0.9329, 0.6441, 0.7710, 0.8494, 0.6686, 0.7434, 0.8922],
    [0.9824, 0.9785, 0.9824, 0.9824, 0.8290, 0.8373, 0.9157, 0.7545, 0.8720, 0.8583],
    [0.9874, 0.8698, 0.9874, 0.9874, 0.7815, 0.8463, 0.8250, 0.7115, 0.8841, 0.8591],
    [0.9807, 0.9177, 0.9807, 0.9807, 0.7450, 0.8556, 0.8587, 0.8459, 0.8841, 0.9064],
    [0.9947, 0.9820, 0.9947, 0.9947, 0.7760, 0.7778, 0.9066, 0.9135, 0.9750, 0.9788]
    # ... 添加更多方法的数据 ...
]).T
# print(performance_matrix.T)
# 执行 Friedman 检验
statistic, p_value = friedmanchisquare(*performance_matrix)

# 输出检验结果
print(f'Friedman Statistic: {statistic}')
print(f'P-value: {p_value}')

# 判断是否拒绝零假设（即，是否存在显著差异）
alpha = 0.05
if p_value < alpha:
    print('拒绝零假设，存在显著差异。')
else:
    print('未能拒绝零假设，不存在显著差异。')

# 进行后面的检验以及画CD图
# 计算每个方法的平均排名
# 使用argsort获取排序后的索引数组
sorted_indices = np.argsort(-performance_matrix, axis=0)
# 使用argsort得到的索引数组构建排名号数组
ranks = np.argsort(sorted_indices, axis=0) + 1
print(ranks)
avranks = np.mean(ranks, axis=1).T
print(avranks)
names = ['DM1', 'DM3', 'DM4', 'DM5', 'DM6', 'DM7', 'DM8', '$Our_{(opt)}$', '$Our_{(com)}$', '$Our_{(pes)}$']
datasets_num = 7
CD = compute_CD(avranks, datasets_num, alpha='0.05', test='nemenyi')
print(f'Nemenyi检验的临界值为{CD:.4f}')
graph_ranks(avranks, names, cd=CD, width=8, textspace=1.5, reverse=True)
plt.show()
