import numpy as np
import matplotlib.pyplot as plt
import math


# normal_distribution

def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)
 

mean1, sigma1 = 0, 1
x1 = np.linspace(mean1 - 6*sigma1, mean1 + 6*sigma1, 100)
 
mean2, sigma2 = 0, 2
x2 = np.linspace(mean2 - 6*sigma2, mean2 + 6*sigma2, 100)
 
mean3, sigma3 = 5, 1
x3 = np.linspace(mean3 - 6*sigma3, mean3 + 6*sigma3, 100)

mean4, sigma4 = -5, 1
x4 = np.linspace(mean4 - 6*sigma4, mean4 + 6*sigma4, 100)
x5 = x1 + x2 + x3 + x4
mean5 = (mean1 + mean2 + mean3 + mean4)/4
sigma5 = (sigma1 + sigma2 + sigma3 + sigma4)/4
y1 = normal_distribution(x1, mean1, sigma1)
y2 = normal_distribution(x2, mean2, sigma2)
y3 = normal_distribution(x3, mean3, sigma3)
y4 = normal_distribution(x4, mean4, sigma4)
y5 = normal_distribution(x5, mean4, sigma4)
# plt.plot(x1, y1, 'r', label='u=0,sig=1')
# plt.plot(x2, y2, 'g', label='u=0,sig=2')
# plt.plot(x3, y3, 'b', label='u=5,sig=1')
# plt.plot(x4, y4, 'y', label='u=-5,sig=1')
plt.plot(x5, y5, 'y', label='u=-5,sig=1')
# plt.scatter(x1,y1,c='r')
# plt.scatter(x2,y2,c='g')
# plt.scatter(x3,y3,c='b')
# plt.scatter(x4,y4,c='y')
plt.scatter(x5,y5,c='b')
plt.legend()
plt.grid()
plt.show()
print("x1-x4 distribution:")
#
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
#
# ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
# #  将数据点分成三部分画，在颜色上有区分度
# ax.scatter(x1[:10], x2[:10], x3[:10], c='y')  # 绘制数据点
# ax.scatter(x1[10:20], x2[10:20], x3[10:20], c='r')
# ax.scatter(x1[30:40], x2[30:40], x3[30:40], c='g')
#
# ax.set_zlabel('Z')  # 坐标轴
# ax.set_ylabel('Y')
# ax.set_xlabel('X')
# mix = np.c_
# plt.show()
# import sys
# sys.argv[0]
#
# def evaluate(x_test,y_test):
#     model = keras.models.load_model(WEIGHT_PATH)
#     model.compile(loss='categorical_crossentrypy',metrics =[metrics.Categorical()])
