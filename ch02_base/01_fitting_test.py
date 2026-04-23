import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # 线性回归模型
from sklearn.preprocessing import PolynomialFeatures # 构建多项式特征
from sklearn.model_selection import train_test_split # 划分训练集和测试集
from sklearn.metrics import mean_squared_error # 均方差误差损失函数

"""
1、生成数据（工程中是导入数据）
2、划分训练集和测试集（验证集）
3、定义模型（线性回归模型）
4、训练模型
5、预测结果，计算误差
"""
plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus']=False

# 1、生成数据（工程中是导入数据）
X = np.linspace(-3,3,300).reshape(-1,1) # -1 是自动计算行数，1 代表 1 列
y = np.sin(X) + np.random.uniform(low=-0.5,high=0.5,size=300).reshape(-1,1) # 生成目标值+噪声 生成 300 个 -0.5 到 0.5 之间的随机数（均匀分布）

print(X.shape)
print(y.shape)
# 画出散点图（3个子图） 创建 1 行 3 列 共 3 个子图 设置总画布大小（宽 15，高 4
# fig 是整张图，ax 是包含 3 个子图的数组
fig, ax = plt.subplots(1,3,figsize=(15,4))
ax[0].scatter(X,y,c='y') # scatter(X,y)：以 X 为横轴、y 为纵轴画散点图
ax[1].scatter(X,y,c='y')
ax[2].scatter(X,y,c='y')
# plt.show()

# 2、划分训练集和测试集（验证集）
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 3、定义模型（线性回归模型）
model = LinearRegression()

# 一、欠拟合
x_train1 = x_train
x_test1 = x_test

# 4、训练模型
model.fit(x_train1,y_train)

# 打印查看模型参数
print(model.coef_) # 斜率 也就是系数
print(model.intercept_) # 截距

# 5、预测结果，计算误差
# 预测测试集 的值
y_test_pred1 = model.predict(x_test1)
# 预测训练集 的值
y_train_pred1 = model.predict(x_train1)

test_loss1 = mean_squared_error(y_test,y_test_pred1)
train_loss1 = mean_squared_error(y_train,y_train_pred1)

# 画出拟合曲线，并写出训练误差和测试误差
ax[0].plot(X,model.predict(X),c='r')
ax[0].text(-3,1,f"测试误差：{test_loss1:.4f}")
ax[0].text(-3,1.3,f"训练误差：{train_loss1:.4f}")

# plt.show()
print('=='*60)
# 二、恰好拟合（5次多项式）
ply5 = PolynomialFeatures(degree=5)
x_train2 = ply5.fit_transform(x_train)
x_test2 = ply5.fit_transform(x_test)

# 4、训练模型
model.fit(x_train2,y_train)

# 打印查看模型参数
print(model.coef_) # 斜率 也就是系数
print(model.intercept_) # 截距

# 5、预测结果，计算误差
y_pred2 = model.predict(x_test2)
test_loss2 = mean_squared_error(y_test,y_pred2)
train_loss2 = mean_squared_error(y_train,model.predict(x_train2))

# 画出拟合曲线，并写出训练误差和测试误差
ax[1].plot(X,model.predict(ply5.fit_transform(X)),c='r')
ax[1].text(-3,1,f"测试误差：{test_loss2:.4f}")
ax[1].text(-3,1.3,f"训练误差：{train_loss2:.4f}")

# plt.show()
print('=='*60)
# 三、过拟合（30次项）
ply30 = PolynomialFeatures(degree=30)
x_train3 = ply30.fit_transform(x_train)
x_test3 = ply30.fit_transform(x_test)

# 4、训练模型
model.fit(x_train3,y_train)

# 打印查看模型参数
print(model.coef_) # 斜率 也就是系数
print(model.intercept_) # 截距

# 5、预测结果，计算误差
y_pred3 = model.predict(x_test3)
test_loss3 = mean_squared_error(y_test,y_pred3)
train_loss3 = mean_squared_error(y_train,model.predict(x_train3))

# 画出拟合曲线，并写出训练误差和测试误差
ax[2].plot(X,model.predict(ply30.fit_transform(X)),c='r')
ax[2].text(-3,1,f"测试误差：{test_loss3:.4f}")
ax[2].text(-3,1.3,f"训练误差：{train_loss3:.4f}")

plt.show()


























