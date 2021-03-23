# By <路畅达&田策文>小组
# 2021.3.23

# 基础函数库
import numpy as np;
import pandas as pd;
#绘图函数库
import matplotlib.pyplot as plt;
import seaborn as sns;
# 我们利用 sklearn 中自带的 iris 数据作为数据载入，并利用Pandas转化为DataFrame格式
from sklearn.datasets import load_iris;

data = load_iris() #得到数据
iris_target = data.target #得到数据对应的标签，其中0，1，2分别代表'setosa', 'versicolor', 'virginica'三种不同花的类别。
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names) 
#data.data得到鸢尾花的数据（也就是花萼和花瓣各自的长宽）
#data.features_names得到data.data中各个数据的名称
#利用Pandas转化为DataFrame格式

# 利用info()查看数据的整体信息
iris_features.info()

# 进行简单的数据查看，我们可以利用 .head() 头部或者.tail()尾部
iris_features.head()

# 利用value_counts函数查看每个类别数量
pd.Series(iris_target).value_counts()

# 对于特征进行一些统计描述
iris_features.describe()

# 合并标签和特征信息
iris_all = iris_features.copy() # 进行浅拷贝，防止对于原始数据的修改
iris_all['target'] = iris_target #给拷贝的数据增加‘target’属性

# 特征与标签组合的散点可视化
sns.pairplot(data=iris_all,diag_kind='hist', hue= 'target')
# diag_kind='hist'设置主对角线为直方图，'kde'设置主对角线为密度图
# hue='target' 设置按照target字段进行分类
plt.show()

for col in iris_features.columns:
    sns.boxplot(x='target', y=col, saturation=0.5,palette='pastel', data=iris_all)
    # x轴为'target'，y为该数据特征的列，saturation为颜色的饱和度，palette为调色板，一共6种，本实验选择的是‘pastel’
    plt.title(col) # 图表名称为数据特征的列
    plt.show()


# 选取其前三个特征绘制三维散点图
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

iris_all_class0 = iris_all[iris_all['target']==0].values
#上述代码得到类别为0的鸢尾花的所有数据，末尾'.values'把数据列表化，前四列为花萼和花瓣各自的长宽，最后一列为鸢尾花的类别
iris_all_class1 = iris_all[iris_all['target']==1].values
iris_all_class2 = iris_all[iris_all['target']==2].values
# 'setosa'(0), 'versicolor'(1), 'virginica'(2)
ax.scatter(iris_all_class0[:,0], iris_all_class0[:,1], iris_all_class0[:,2],label='setosa')
#只用到前三列
ax.scatter(iris_all_class1[:,0], iris_all_class1[:,1], iris_all_class1[:,2],label='versicolor')
ax.scatter(iris_all_class2[:,0], iris_all_class2[:,1], iris_all_class2[:,2],label='virginica')
plt.legend() # 显示图中点代表的类别

plt.show()




