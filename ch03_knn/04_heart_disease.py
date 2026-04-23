import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # 设置为你想使用的核心数


import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV # 数据集划分 和 网格搜索
from sklearn.preprocessing import OneHotEncoder,StandardScaler # 独热编码 和 标准化
from sklearn.compose import ColumnTransformer # 列转换器 也可以用 管道
from sklearn.neighbors import KNeighborsClassifier # knn分类模型
import joblib


# 加载数据
heart_disease_data = pd.read_csv(filepath_or_buffer="../data/heart_disease.csv",sep=',',encoding='utf-8')


# 缺失值处理
heart_disease_data.dropna(how='any',axis=0,inplace=True) # 原地修改
heart_disease_data.info()
print(heart_disease_data.head())
print(heart_disease_data.shape)

# 数据集划分
X = heart_disease_data.drop("是否患有心脏病",axis=1) # 特征
y = heart_disease_data["是否患有心脏病"] # 标签
print(X.shape, y.shape) # (1025, 13) (1025,)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
# =======================================================================================================================
"""
    特征工程
        此数据集有多种数据，已经进行了数值化了特征，但是类别型特征是相互独立，跟他们之间的距离没有关系
        故而 对于类别型特征 需要进行独热编码处理 >> 独热编码之后删除第一列 >> 避免多重共线性
        
    使用了列转换器 >> 方便多列进行不同的特征处理
"""
# 特征工程
# 类别型特征
categorical_features = ["胸痛类型", "静息心电图结果", "峰值ST段的斜率", "地中海贫血"]
# 数值型特征
numerical_features = ["年龄","静息血压","胆固醇","最大心率","运动后的ST下降","主血管数量"]
# 二元型特征
binary_features = ["性别", "空腹血糖", "运动性心绞痛"]
# 创建 列转换器
columnTransformer = ColumnTransformer(
    transformers=[
        # 类别型特征处理 独热编码 使用drop="first" 表示去掉一个特征 避免多重共线性
        ('cat', OneHotEncoder(drop="first"), categorical_features),
        # 数值型特征处理 标准化
        ('num', StandardScaler(), numerical_features),
        # 二元型特征处理 不做处理
        ('bin', 'passthrough', binary_features)
    ]
)
# 执行特征转换
# 训练集 使用fit_transform 可以计算出 标准差，均值 让测试集也使用
X_train = columnTransformer.fit_transform(X_train)
# 测试集 使用transform 只使用已经计算好的 标准差，均值
X_test = columnTransformer.transform(X_test)
# =======================================================================================================================
"""
    简单流程
        1、创建模型
        2、模型训练
        3、模型评估
        4、模型保存
    总结：这种方式执行一次，只能验证一次 超参数 的模型训练结果
"""
# =======================================================================================================================
# # 创建模型
# knn = KNeighborsClassifier(n_neighbors=3)
#
# # 模型训练
# knn.fit(X_train,y_train)
#
# # 模型评估
# # y_pred = knn.predict(X_test)
# score = knn.score(X_test,y_test)
# print("准确率：",score)

# # 模型保存 如果没有指定扩展名，joblib 会自动添加 .pkl 扩展名
# joblib.dump(value=knn,filename="../model/heart_disease")
# =======================================================================================================================

# # 模型加载
# model = joblib.load(filename="../model/heart_disease")
# y_pred = model.predict(X_test[10:11])
# print("准确率：",model.score(X_test,y_test))
# print(f"测试集预测结果：{y_pred},真实值：{y_test[10]}")

# =======================================================================================================================
"""
    模型交叉验证 >> 网格搜索进行交叉验证
        1、创建模型（不加参数）
        2、配置模型参数字典（包含了对应模型的相关超参数）
        3、创建网格搜索对象（将模型对象，超参数，k折处理相关信息）
        4、模型训练，这里使用了网格搜索对象进行模型训练（里面包含了模型对象），对模型根据不同参数组合进行训练
        5、模型评估（生成不同组合参数训练出来的模型性能进行评估和排名）
"""
# =======================================================================================================================

# 模型交叉验证  网格搜索参数，K值设置为1到10
# # 创建模型
# knn = KNeighborsClassifier()
# # 模型参数 n_neighbors K值 weights 权重
# param_grid = {"n_neighbors":list(range(1,11)),"weights":["uniform","distance"]}
#
# # 创建网格搜索对象 参数estimator:模型，param_grid:参数 cv:交叉验证次数(训练集被拆分为10折进行交叉验证)
# grid_search_cv = GridSearchCV(estimator=knn,param_grid=param_grid,cv=10)
#
# # 模型训练
# grid_search_cv.fit(X_train,y_train)
#
# results = pd.DataFrame(grid_search_cv.cv_results_).to_string()
# print(results)
# print(type(results))
# print("最佳参数：",grid_search_cv.best_params_)
# print("最佳模型：",grid_search_cv.best_estimator_)
# print("最佳模型得分：",grid_search_cv.best_score_)
#
# # 使用最佳模型进行评估
# knn = grid_search_cv.best_estimator_
# print("准确率：",knn.score(X_test,y_test))
#
# # 最佳模型保存
# joblib.dump(value=knn,filename="../model/heart_disease_grid_search")
# =======================================================================================================================

# 加载网格搜索最佳模型并且进行验证
model_knn = joblib.load(filename="../model/heart_disease_grid_search")
print("准确率：",model_knn.score(X_test,y_test))
print(f"测试集预测结果：{model_knn.predict(X_test[10:11])},真实值：{y_test[10]}")
# =======================================================================================================================












