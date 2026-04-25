"""
    朴素贝叶斯算法实现
        朴素贝叶斯 - 高斯模型（连续数据）
    三种贝叶斯
        GaussianNB：连续特征（默认）
        MultinomialNB：文本 / 计数特征
        BernoulliNB：0-1 二值特征
"""
from sklearn.datasets import load_iris # 导入数据集
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# from sklearn.pipeline import Pipeline


# 1- 加载数据
dataset = load_iris()
X = dataset.data
y = dataset.target

# 2- 划分数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2026,shuffle= True)

# 3- 创建模型
model = GaussianNB()

# 4- 模型训练
model.fit(X_train,y_train)

# 5- 模型评估
y_pred = model.predict(X_test)
print("准确率：",accuracy_score(y_test,y_pred))