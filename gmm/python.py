# _*_coding:utf-8_*_
# create by ChuangweiZhu on 2021/5/6 21:25
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
import os
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
n_samples = 800
np.random.seed(13456)

''' 使用多个正态分布生成数据'''
# shifted_gaussian = np.random.randn(n_samples, 2) + np.array([10, 20])
# C = np.array([[0., -0.7], [5, .7]])
# stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
# concatenate the two datasets into the final training set
#X_train = np.vstack([shifted_gaussian, stretched_gaussian])

''' 加载现有数据文本   '''
Y = np.loadtxt("gmm.data")
X_train = np.matrix(Y, copy=True)
X_train = np.array(X_train)
''' 显示高数斯密度曲线  '''
def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)


x_mean = np.mean(X_train[:, 0])

# 拟合数据聚簇两类
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)

# 等高线展示
x = np.linspace(1.5, 5.)
y = np.linspace(32., 100.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], c='r', alpha=0.4)

# plt.title(' Sigal GM ')
plt.title(' GMM ')

plt.axis('tight')
plt.show()