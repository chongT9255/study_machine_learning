"""
    AdaBoost 实现

"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier # AdaBoost
from sklearn.tree import DecisionTreeClassifier # 决策树作为 基学习器
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# 解决matplotlib中文乱码
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
# 1- 加载数据
dataset = load_iris()
X = dataset.data
y = dataset.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2026,shuffle= True)
print("数据集大小：",X.shape,y.shape)
print("训练集大小：",X_train.shape,y_train.shape)
print("测试集大小：",X_test.shape,y_test.shape)
# # 2- 创建模型: 基学习器=单层决策树
# ada = AdaBoostClassifier(
#     estimator=DecisionTreeClassifier(max_depth=1), # 基学习器
#     n_estimators=50, # 基学习器数量
#     # learning_rate=0.5, # 基学习器权重
#     # algorithm="SAMME"
# )
# # 3- 模型训练
# ada.fit(X_train,y_train)
# y_pred = ada.predict(X_test)
# print("准确率：",accuracy_score(y_test,y_pred))
# 2. 训练不同数量弱分类器的AdaBoost
n_estimators_range = range(1, 51)
train_accs = []
test_accs = []

for n in n_estimators_range:
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n,
        random_state=42
    )
    ada.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, ada.predict(X_train)))
    test_accs.append(accuracy_score(y_test, ada.predict(X_test)))
# 3 可视化准确率变化
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_accs, label="训练集准确率", marker='o')
plt.plot(n_estimators_range, test_accs, label="测试集准确率", marker='s')
plt.xlabel("弱分类器数量 (n_estimators)")
plt.ylabel("准确率")
plt.title("AdaBoost 准确率随弱分类器数量的变化")
plt.legend()
plt.grid(True)
plt.show()