import numpy as np

# 定义目标函数（损失函数）
def J(beta):
    # X 是 n行2列
    # beta 是 2行1列 ==》 beta0 合 beta1 一元线性方程组
    # 得到得结果是：n 行 1列
    return np.sum( (X @ beta - y) ** 2) / n

# 定义梯度函数
def gradient(beta):
    # X 是 n行2列
    # beta 是 2行1列 ==》 beta0 合 beta1 一元线性方程组
    # 梯度矩阵是 2行1列
    return X.T @ (X @ beta - y) / n * 2
sdsd

# 1-定义数据
X = np.array([[5], [8], [10], [12], [15], [3], [7], [9], [14], [6]])  # 自变量，每周学习时长
y = np.array([[55], [65], [70], [75], [85], [50], [60], [72], [80], [58]]) # 因变量，数学考试成绩

n = X.shape[0] # 得到多少行

# 2- 数据处理，X增加一列
X = np.hstack([np.ones((n,1)),X])

# 3- 定义初始参数
beta = np.array([[1], [1]]) # 初始参数 2行1列
alpha = 0.01 # 学习率
iter = 10000

beta0 = []
beta1 = []
# 重复迭代
# for i in range(iter):# 迭代次数
while np.any(np.abs(grad := gradient(beta)) > 1e-10) and (iter := iter - 1) >= 0:
    # grad = gradient(beta)
    beta0.append(beta[0][0])
    beta1.append(beta[1][0])
    beta = beta - alpha * grad

    if iter % 10 == 0:
        print(f"迭代次数：{10000 - iter}，参数：{beta.reshape(-1)}，损失函数：{J(beta)}")


# 画图
import matplotlib.pyplot as plt
plt.plot(beta0,beta1)
plt.show()
