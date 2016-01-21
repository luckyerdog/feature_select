# -*- coding: utf-8 -*-

#一个完整的比较例

from sklearn.datasets import load_boston
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from minepy import MINE
np.random.seed(0)
size = 750
X = np.random.uniform(0, 1, (size, 14))
#"Friedamn #1” regression problem
Y = (10 * np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2] - .5)**2 + 10*X[:,3] + 5*X[:,4] + np.random.normal(0,1))
#Add 4 additional correlated variables (correlated with X1-X4)
X[:,10:] = X[:,:4] + np.random.normal(0, .025, (size,4))
names = ["x%s" % i for i in range(1,15)]
ranks = {}
def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

#2.1 线性模型
lr = LinearRegression(normalize=True)
lr.fit(X, Y)
ranks["L_reg"] = rank_to_dict(np.abs(lr.coef_), names)

#2.2 正则化Ridge
ridge = Ridge(alpha=7)
ridge.fit(X, Y)
ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)

#2.2 正则化Lasso
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)

#4.1 顶层特征选择->稳定性选择->随机lasso
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)

#4.2 顶层特征选择->递归特征消除
#stop the search when 5 features are left (they will get equal scores)
rfe = RFE(lr, n_features_to_select=5)
rfe.fit(X,Y)
ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), names, order=-1)

#3.1 随机森林->平均不纯度减少
rf = RandomForestRegressor()
rf.fit(X,Y)
ranks["RF"] = rank_to_dict(rf.feature_importances_, names)

#2.1 单变量的线性回归
f, pval = f_regression(X, Y, center=True)
ranks["Corr."] = rank_to_dict(f, names)

#1.2 单变量特征选择->互信息和最大信息系数
mine = MINE()
mic_scores = []
for i in range(X.shape[1]):
    mine.compute_score(X[:,i], Y)
    m = mine.mic()
    mic_scores.append(m)
ranks["MIC"] = rank_to_dict(mic_scores, names)

#算出各变量系数的平均值
r = {}
for name in names:
    r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)
    
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")

print "\t%s" % "\t".join(methods)  #输出表头

#输出各变量在各个方法下的特征系数
for name in names:
    print "%s\t%s" % (name, "\t".join(map(str, [ranks[method][name] for method in methods])))


# 从输出结果中可以找到一些有趣的发现：

# -*- 特征之间存在线性关联关系，每个特征都是独立评价的，因此X1,…X4的得分和X11,…X14的得分非常接近，
# -*- -*- 而噪音特征X5,…,X10正如预期的那样和响应变量之间几乎没有关系。由于变量X3是二次的，因此X3和响应变量之间
# -*- -*- 看不出有关系（除了MIC之外，其他方法都找不到关系）。这种方法能够衡量出特征和响应变量之间的线性关系，但若想选出优质
# -*- -*- 特征来提升模型的泛化能力，这种方法就不是特别给力了，因为所有的优质特征都不可避免的会被挑出来两次。

# -*- Lasso能够挑出一些优质特征，同时让其他特征的系数趋于0。当如需要减少特征数的时候它很有用，但是对于数据理解
# -*- -*- 来说不是很好用。（例如在结果表中，X11,X12,X13的得分都是0，好像他们跟输出变量之间没有很强的联系，但实际上不是这样的）

# -*- MIC对特征一视同仁，这一点上和关联系数有点像，另外，它能够找出X3和响应变量之间的非线性关系。

# -*- 随机森林基于不纯度的排序结果非常鲜明，在得分最高的几个特征之后的特征，得分急剧的下降。从表中可以看到，得分
# -*- -*- 第三的特征比第一的小4倍。而其他的特征选择算法就没有下降的这么剧烈。

# -*- Ridge将回归系数均匀的分摊到各个关联变量上，从表中可以看出，X11,…,X14和X1,…,X4的得分非常接近。

# -*- 稳定性选择常常是一种既能够有助于理解数据又能够挑出优质特征的这种选择，在结果表中就能很好的看出。
# -*- -*- 像Lasso一样，它能找到那些性能比较好的特征（X1，X2，X4，X5），同时，与这些特征关联度很强的变量也得到了较高的得分。

#总结
# -*- 对于理解数据、数据的结构、特点来说，单变量特征选择是个非常好的选择。尽管可以用它对特征进行排序来优化模型，
# -*- -*- 但由于它不能发现冗余（例如假如一个特征子集，其中的特征之间具有很强的关联，那么从中选择最优的特征时就很难考虑到
# -*- -*- 冗余的问题）。

# -*- 正则化的线性模型对于特征理解和特征选择来说是非常强大的工具。L1正则化能够生成稀疏的模型，对于选择特征子集来说
# -*- -*- 非常有用；相比起L1正则化，L2正则化的表现更加稳定，由于有用的特征往往对应系数非零，因此L2正则化对于数据的理解来说
# -*- -*- 很合适。由于响应变量和特征之间往往是非线性关系，可以采用basis expansion的方式将特征转换到一个更加合适的空间当中，
# -*- -*- 在此基础上再考虑运用简单的线性模型。

# -*- 随机森林是一种非常流行的特征选择方法，它易于使用，一般不需要feature engineering、调参等繁琐的步骤，并且
# -*- -*- 很多工具包都提供了平均不纯度下降方法。它的两个主要问题，1是重要的特征有可能得分很低（关联特征问题），2是这种方法
# -*- -*- 对特征变量类别多的特征越有利（偏向问题）。尽管如此，这种方法仍然非常值得在你的应用中试一试。

# -*- 特征选择在很多机器学习和数据挖掘场景中都是非常有用的。在使用的时候要弄清楚自己的目标是什么，然后找到哪种方法
# -*- -*- 适用于自己的任务。当选择最优特征以提升模型性能的时候，可以采用交叉验证的方法来验证某种方法是否比其他方法要好。
# -*- -*- 当用特征选择的方法来理解数据的时候要留心，特征选择模型的稳定性非常重要，稳定性差的模型很容易就会导致错误的结论。
# -*- -*- 对数据进行二次采样然后在子集上运行特征选择算法能够有所帮助，如果在各个子集上的结果是一致的，那就可以说在
# -*- -*- 这个数据集上得出来的结论是可信的，可以用这种特征选择模型的结果来理解数据。
