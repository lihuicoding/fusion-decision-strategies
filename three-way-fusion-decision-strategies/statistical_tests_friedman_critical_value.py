from scipy.stats import f
# 数据集个数
n = 7
# 算法个数
k = 10
# 显著性
alpha = 0.05
# friedman test临界值
critical_value = f.ppf(q=1-alpha, dfn=k-1, dfd=(k-1)*(n-1))
print(f'Friedman检验的临界值为{critical_value:.4f}')
