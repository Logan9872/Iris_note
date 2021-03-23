# By <路畅达&田策文>小组
# 2021.3.22

# 引入数据库
import pandas as pd;
import numpy as np;
import mglearn;
import matplotlib.pyplot as plt;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.datasets import load_iris;
from sklearn.model_selection import train_test_split;

# 1数据库信息
iris_dataset = load_iris();
print(iris_dataset['DESCR'][:193] + "n\...");
print("------------------------------------------------------------------------------------------------------------------------------")
print("1.1数据组键名:{}".format(iris_dataset.keys()));
print("1.2.预测花品种:{}".format(iris_dataset['target_names']));
print("1.3特征名称:{}".format(iris_dataset['feature_names']));
print("1.4数据类型:{}".format(type(iris_dataset['data'])));
print("1.5数据格式:{}".format(iris_dataset['data'].shape));
print("1.6数据前5行:\n{}".format(iris_dataset['data'][:5]));
print("1.7目标数据类型:{}".format(type(iris_dataset['target'])));
print("1.8目标数据格式:{}".format(iris_dataset['target'].shape));
print("1.9目标:\n{}".format(iris_dataset['target']));
print("------------------------------------------------------------------------------------------------------------------------------")

#2.训练集&测试集
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0);
print("2.1数据X训练集格式:{}".format(X_train.shape));
print("2.2标签y训练集格式:{}".format(y_train.shape));
print("2.3数据X测试集格式:{}".format(X_train.shape));
print("2.4标签y测试集格式:{}".format(y_test.shape));
print("------------------------------------------------------------------------------------------------------------------------------")

# 数据可视化

# 用X_train创建数据库，并对数据列进行标记
iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(
    iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
    hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3);
plt.show()


# k临近算法
knn = KNeighborsClassifier(n_neighbors=1);
knn.fit(X_train, y_train);

#3.做出预测
X_new = np.array([[5, 2.9, 1, 0.2]]);
print("3.1X_new的格式:{}".format(X_new.shape));
#调用knn对象的predict方法进行预测
prediction = knn.predict(X_new);
print("3.2Prediction:{}".format(prediction));
print("3.3预测目标姓名:{}".format(
    iris_dataset['target_names'][prediction]
));
print("------------------------------------------------------------------------------------------------------------------------------")

#评估模型
y_pred = knn.predict(X_test);
print("4.1测试集预测结果:\n{}".format(y_pred));
print("4.2测试集得分:{:.2f}".format(np.mean(y_pred == y_test)));
print("4.3测试集得分:{:.2f}".format(knn.score(X_test,y_test)));
print("------------------------------------------------------------------------------------------------------------------------------")

