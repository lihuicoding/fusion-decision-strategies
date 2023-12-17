import numpy as np
from scipy.stats import friedmanchisquare
# import Orange  # version: Orange3==3.32.0
from Orange.evaluation.scoring import compute_CD
from Orange.evaluation.scoring import graph_ranks
import matplotlib.pyplot as plt

# 创建一个示例的性能指标矩阵，每行代表一个方法，每列代表一个数据集
# 10个方法，7个数据集的示例数据
performance_matrix = np.array([
    [0.9861, 0.9850, 0.9708, 0.9982, 0.9989, 0.9974, 0.9998],
    [0.9629, 0.8842, 0.9047, 0.9976, 0.9073, 0.9626, 0.9981],
    [0.9861, 0.9850, 0.9708, 0.9982, 0.9989, 0.9974, 0.9998],
    [0.9861, 0.9850, 0.9708, 0.9982, 0.9989, 0.9974, 0.9998],
    [0.8499, 0.5882, 0.4933, 0.8809, 0.7983, 0.7354, 0.7917],
    [0.9191, 0.7558, 0.7997, 0.8904, 0.8893, 0.9064, 0.7949],
    [0.8761, 0.8797, 0.8808, 0.9574, 0.8002, 0.8970, 0.9516],
    [0.9499, 0.6533, 0.5027, 0.6945, 0.5052, 0.8658, 0.9089],
    [0.9736, 0.6939, 0.6725, 0.9109, 0.9285, 0.9223, 0.9939],
    [0.9743, 0.9500, 0.9341, 0.8834, 0.8789, 0.9420, 0.9951]
    # ... 添加更多方法的数据 ...
])

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
