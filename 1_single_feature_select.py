# -*- coding: utf-8 -*-

#1 单变量特征选择

#1.1 Pearson相关系数 （Pearson Correlation) [-1,1]
#理解特征和响应变量之间关系的方法，该方法衡量的是变量之间的线性相关性

import numpy as np
from scipy.stats import pearsonr  #从scipy中引入pearsonr
np.random.seed(0)
size = 300
x = np.random.normal(0, 1, size)  #normal(mean,stdev,size) 高斯数
print "Lower noise", pearsonr(x, x + np.random.normal(0, 1, size))
print "Higher noise", pearsonr(x, x + np.random.normal(0, 10, size))

#明显缺陷:作为特征排序机制，他只对线性关系敏感.即便两个变量具有一一对应的关系，Pearson相关性也可能会接近0
a = np.random.uniform(-1, 1, 100000)   #uniform(low,high,size) 随机数
print pearsonr(a, a**2)[0]


#1.2 互信息和最大信息系数 (Mutual information and maximal information)，[0,1]
#互信息直接用于特征选择不太方便，最大信息系数首先寻找一种最优的离散化方式，
#然后把互信息取值转换成一种度量方式，取值区间在[0，1]。minepy提供了MIC功能。

from minepy import MINE  #
m = MINE()
x = np.random.uniform(-1, 1, 10000)
m.compute_score(x, x**2)
print m.mic()


#1.3 距离相关系数 (Distance correlation)，[0,1]
#距离相关系数是为了克服Pearson相关系数的弱点而生的。在x和x^2这个例子中，即便Pearson相关系数是0，
#我们也不能断定这两个变量是独立的（有可能是非线性相关）；但如果距离相关系数是0，那么我们就可以说这两个变量是独立的。
import numpy as np

def dist(x, y):
    #1d only
    return np.abs(x[:, None] - y)
    

def d_n(x):
    d = dist(x, x)
    dn = d - d.mean(0) - d.mean(1)[:,None] + d.mean()
    return dn


def dcov_all(x, y):
    dnx = d_n(x)
    dny = d_n(y)
    
    denom = np.product(dnx.shape)
    dc = (dnx * dny).sum() / denom
    dvx = (dnx**2).sum() / denom
    dvy = (dny**2).sum() / denom
    dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
    return dc, dr, dvx, dvy


import matplotlib.pyplot as plt

fig = plt.figure()
for case in range(1,5):

    np.random.seed(9854673)
    x = np.linspace(-1,1, 501)
    if case == 1:
        y = - x**2 + 0.2 * np.random.rand(len(x))
    elif case == 2:
        y = np.cos(x*2*np.pi) + 0.1 * np.random.rand(len(x))
    elif case == 3:
        x = np.sin(x*2*np.pi) + 0.0 * np.random.rand(len(x))  #circle
    elif case == 4:
        x = np.sin(x*1.5*np.pi) + 0.1 * np.random.rand(len(x))  #bretzel
    dc, dr, dvx, dvy = dcov_all(x, y)
    print dc, dr, dvx, dvy
    
    ax = fig.add_subplot(2,2, case)
    #ax.set_xlim(-1, 1)
    ax.plot(x, y, '.')
    yl = ax.get_ylim()
    ax.text(-0.95, yl[0] + 0.9 * np.diff(yl), 'dr=%4.2f' % dr)

plt.show()


#1.4 基于学习模型的特征排序 (Model based ranking)
#直接使用你要用的机器学习算法，针对每个单独的特征和响应变量建立预测模型。线性：pearson等价；非线性：基于树的方法（决策树、随机森林）
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
#Load boston housing dataset as an example
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]
rf = RandomForestRegressor(n_estimators=20, max_depth=4)  #树的深度最好不要太大，再就是运用交叉验证
scores = []
for i in range(X.shape[1]):
    score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2", cv=ShuffleSplit(len(X), 3, .3))
    scores.append((round(np.mean(score), 3), names[i]))
print sorted(scores, reverse=True)

