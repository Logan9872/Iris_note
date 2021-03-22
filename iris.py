# By <路畅达&田策文>小组
# 2021.3.22

from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("key of iris_dataset:\n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193]+"n\...")
