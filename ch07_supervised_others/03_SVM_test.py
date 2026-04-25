"""
    支持向量机实现
        # 线性核（数据线性可分时用）
        SVC(kernel='linear')

        # 多项式核（复杂非线性）
        SVC(kernel='poly')

        # 高斯核 RBF（最常用、默认、效果最好）
        SVC(kernel='rbf')

        # Sigmoid 核（类似神经网络）
        SVC(kernel='sigmoid')
    关键参数：
    C：越大，越在意错分样本（容易过拟合）
    gamma：越大，模型越复杂（过拟合风险↑）
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC,SVR
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
# gamma = 1 / (n_features * X.var())
model = SVC(C=1.0,kernel="rbf",gamma="scale",random_state=2026)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("准确率：",accuracy_score(y_test,y_pred))
print("支持向量：",model.support_vectors_)