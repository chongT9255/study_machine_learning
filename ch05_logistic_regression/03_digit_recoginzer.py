import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler # 归一化
from sklearn.linear_model import LogisticRegression # 逻辑回归


# 1- 加载数据集
dataset = pd.read_csv('../data/train.csv')

# 测试图像
# digit = dataset.iloc[110,1:].values # <class 'pandas.core.series.Series'>
# plt.imshow(digit.reshape(28,28),cmap='gray')
# plt.show()

# 2- 划分数据集
# 方式一：
# X = dataset.iloc[:,1:]
# y = dataset.iloc[:,0]
# print(X.shape, y.shape)
# print(X.head())
# print(y.head())
# 方式二：
X = dataset.drop('label',axis=1)
y = dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 3- 特征工程：进行归一化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape)

# 4- 模型创建：逻辑回归分类模型
model = LogisticRegression(max_iter=500)

# 5- 模型训练
model.fit(X_train, y_train)

# 6- 模型评估
score = model.score(X_test, y_test)
print(score) # 0.919047619047619

# 7- 测试（预测某个图像表示的数字）
print(type(X_test)) # <class 'numpy.ndarray'>
print(type(y_test)) # <class 'pandas.core.series.Series'>
digit = X_test[101,:].reshape(1,-1) # 转化为1*784维
y_pred = model.predict(digit)

print(f"预测结果：{y_pred},真实值：{y_test.iloc[101]}")

# 画出图像
plt.imshow(digit.reshape(28,28),cmap='gray')
plt.show()