# By <路畅达&田策文>小组
# 2021.3.22

# 引入数据库
import pandas as pd;
import numpy as np;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.datasets import load_iris;
from sklearn.model_selection import train_test_split;

# 数据库信息
iris_dataset = load_iris();
print(iris_dataset['DESCR'][:193] + "n\...");
print("##############################################################################################################################")
print("1.数据组键名:{}".format(iris_dataset.keys()));
print("2.预测花品种:{}".format(iris_dataset['target_names']));
print("3.特征名称:{}".format(iris_dataset['feature_names']));
print("4.数据类型:{}".format(type(iris_dataset['data'])));
print("5.数据格式:{}".format(iris_dataset['data'].shape));
print("6.数据前5行:\n{}".format(iris_dataset['data'][:5]));
print("7.目标数据类型:{}".format(type(iris_dataset['target'])));
print("8.目标数据格式:{}".format(iris_dataset['target'].shape));
print("9.目标:\n{}".format(iris_dataset['target']));
print("##############################################################################################################################")

#训练集&测试集
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0);
print("1.数据X训练集格式:{}".format(X_train.shape));
print("2.标签y训练集格式:{}".format(y_train.shape));
print("3.数据X测试集格式:{}".format(X_train.shape));
print("4.标签y测试集格式:{}".format(y_test.shape));
print("##############################################################################################################################")

# 数据可视化

# 用X_train创建数据库，并对数据列进行标记
# iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)

# grr = pd.scatter_matrix(
#     iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
#     hist_kwds={'bins':20},s=60,alpha=.8, cmap=mglearn.cm3
# )

# k临近算法
knn = KNeighborsClassifier(n_neighbors=1);
knn.fit(X_train, y_train);

#做出预测
x_new = np.array([5, 2.9, 1, 0.2]);
print("X_new的格式:{}".format(x_new.shape));