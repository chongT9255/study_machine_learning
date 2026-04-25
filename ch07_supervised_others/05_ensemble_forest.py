"""
    随机森林实现
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
# 解决matplotlib中文乱码
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
# 1- 加载数据
dataset = load_iris()
X = dataset.data
y = dataset.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2026,shuffle= True)

# 2- 创建模型
model = RandomForestClassifier(
    n_estimators=100, # 树的数量
    max_features="sqrt", # 随机选取特征，RF核心：特征数量sqrt(n_features)
    # max_depth=4, # 树的最大深度
    n_jobs=-1, # 使用CPU核数 并行
    random_state=2026
)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("准确率：",accuracy_score(y_test,y_pred))

# 3. 可视化随机森林中的第1棵树
plt.figure(figsize=(12, 8))
plot_tree(
    model.estimators_[0],  # 取第一棵树
    feature_names=dataset.feature_names,
    class_names=dataset.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("随机森林 - 第1棵决策树可视化")
plt.show()