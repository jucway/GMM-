import matplotlib.pyplot as plt
from gmm import *

# 设置调试模式
DEBUG = True

# 载入数据
Y = np.loadtxt("gmm.data")
matY = np.matrix(Y, copy=True)

#
# x = Y[:, 0]
# y = Y[:, 1]
#
#
#
# plt.title('data distribution ', fontsize=22)
# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)
# axes = plt.subplot(111)
#
# plt.plot(x, y, "r*")
# plt.show()
#









# 模型个数，即聚类的类别个数
K = 2

# 计算 GMM 模型参数
mu, cov, alpha = GMM_EM(matY, K, 100)

# 根据 GMM 模型，对样本数据进行聚类，一个模型对应一个类别
N = Y.shape[0]
# 求当前模型参数下，各模型对样本的响应度矩阵
gamma = getExpectation(matY, mu, cov, alpha)
# 对每个样本，求响应度最大的模型下标，作为其类别标识
category = gamma.argmax(axis=1).flatten().tolist()[0]
# 将每个样本放入对应类别的列表中
class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
class2 = np.array([Y[i] for i in range(N) if category[i] == 1])

# 绘制聚类结果
plt.plot(class1[:, 0], class1[:, 1], 'rs',alpha=0.4, label="class1")
plt.plot(class2[:, 0], class2[:, 1], 'bs', alpha=0.4,label="class2")
plt.legend(loc="best")
plt.title("GMM Clustering By EM Algorithm")
plt.show()


