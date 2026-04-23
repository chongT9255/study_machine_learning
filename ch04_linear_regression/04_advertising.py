import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # 标准化
from sklearn.metrics import mean_squared_error
# 导入数据
dataset = pd.read_csv('../data/advertising.csv')

dataset.dropna(axis=0,inplace=True)
dataset.drop(columns=dataset.columns[0],inplace=True,axis=1)
dataset.info()
print(dataset.head())

# 数据集划分
X = dataset.drop(columns=['Sales'])
y = dataset['Sales']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# 特征工程
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建模型
model_lr = LinearRegression() # 正规方程法
model_sgd = SGDRegressor() # 随机梯度下降法

# 模型训练
model_lr.fit(X_train,y_train)
model_sgd.fit(X_train,y_train)

# 打印系数和截距
print("lr模型系数：",model_lr.coef_)
print("lr模型截距：",model_lr.intercept_)
print("sgd模型系数：",model_sgd.coef_)
print("sgd模型截距：",model_sgd.intercept_)

# 模型预测
y_pred_lr = model_lr.predict(X_test)
y_pred_sgd = model_sgd.predict(X_test)
# 模型评估
print("lr模型得分：",mean_squared_error(y_pred_lr,y_test))
print("sgd模型得分：",mean_squared_error(y_pred_sgd,y_test))