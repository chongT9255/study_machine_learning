"""
    决策树实现（CART决策树）
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1- 加载数据
dataset = load_iris()
X = dataset.data
y = dataset.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2026,shuffle= True)
print("数据集大小：",X.shape,y.shape)
print("训练集大小：",X_train.shape,y_train.shape)
print("测试集大小：",X_test.shape,y_test.shape)
# 2- 创建模型
model = DecisionTreeClassifier(criterion="gini",max_depth=4,random_state=2026,min_samples_split=2)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("准确率：",accuracy_score(y_test,y_pred))
print("特征重要性：",model.feature_importances_)
print("决策树结构：",model.tree_)

# 可视化决策树
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
plot_tree(model, filled=True, feature_names=dataset.feature_names, class_names=dataset.target_names, rounded=True)
plt.show()