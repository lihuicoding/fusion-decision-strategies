import scipy.stats
import math

# 假设 n 为样本量
n = 4177

# 计算自由度
df = n - 2

# 设定显著性水平
alpha = 0.05

# 计算单尾检验（正相关）的临界值
critical_value = scipy.stats.t.ppf(1 - alpha, df)
print("单尾检验的临界值（正相关）:", critical_value)
# 计算对应的斯皮尔曼临界值
srcc = critical_value / math.sqrt(n - 2 + critical_value ** 2)
print('srcc值', srcc)
