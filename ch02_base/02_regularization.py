import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Lasso,Ridge # 线性回归模型,lasso回归，岭回归
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

# 画出散点图（3个子图） 2*3的图像
fig, ax = plt.subplots(2,3,figsize=(15,4))
ax[0,0].scatter(X,y,c='y') # scatter(X,y)：以 X 为横轴、y 为纵轴画散点图
ax[0,1].scatter(X,y,c='y')
ax[0,2].scatter(X,y,c='y')
# plt.show()

# 2、划分训练集和测试集（验证集）
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# 过拟合情况（30次项）
ply30 = PolynomialFeatures(degree=30)
x_train = ply30.fit_transform(x_train)
x_test = ply30.fit_transform(x_test)

# 一、不加正则化
# 3、定义模型
model = LinearRegression()

# 4、训练模型
model.fit(x_train,y_train)

# 打印查看模型参数
print(model.coef_) # 斜率 也就是系数
print(model.intercept_) # 截距

# 5、预测结果，计算误差
y_pred1 = model.predict(x_test)
test_loss1 = mean_squared_error(y_test,y_pred1)

# 画出拟合曲线，并写出训练误差和测试误差
ax[0,0].plot(X,model.predict(ply30.fit_transform(X)),c='r')
ax[0,0].text(-3,1,f"测试误差：{test_loss1:.4f}")
ax[0,0].text(-3,1.3,"不加正则化")
# 系数直方图
ax[1,0].bar(np.arange(31),model.coef_.reshape(-1))

# 二、加Lasso正则化（Lasso回归）
# 3、定义模型
lasso = Lasso(alpha=0.01)

# 4、训练模型
lasso.fit(x_train,y_train)

# 5、预测结果，计算误差
y_pred2 = lasso.predict(x_test)
test_loss2 = mean_squared_error(y_test,y_pred2)

# 画出拟合曲线，并写出训练误差和测试误差
ax[0,1].plot(X,lasso.predict(ply30.fit_transform(X)),c='r')
ax[0,1].text(-3,1,f"测试误差：{test_loss2:.4f}")
ax[0,1].text(-3,1.3,"Lasso回归")
# 系数直方图
ax[1,1].bar(np.arange(31),lasso.coef_.reshape(-1))



# 三、加岭回归正则化（岭回归）

# 3、定义模型
ridge = Ridge(alpha=1)

# 4、训练模型
ridge.fit(x_train,y_train)

# 5、预测结果，计算误差
y_pred3 = ridge.predict(x_test)
test_loss3 = mean_squared_error(y_test,y_pred3)

# 画出拟合曲线，并写出训练误差和测试误差
ax[0,2].plot(X,ridge.predict(ply30.fit_transform(X)),c='r')
ax[0,2].text(-3,1,f"测试误差：{test_loss3:.4f}")
ax[0,2].text(-3,1.3,"Ridge回归")
# 系数直方图
ax[1,2].bar(np.arange(31),ridge.coef_.reshape(-1))


plt.show()


























