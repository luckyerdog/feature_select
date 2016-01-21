# -*- coding: utf-8 -*-

#4 两种顶层特征选择算法

#4.1 稳定性选择 (Stability selection)  [0,1]
#它的主要思想是在不同的数据子集和特征子集上运行特征选择算法，不断的重复，最终汇总特征选择结果，
#比如可以统计某个特征被认为是重要特征的频率（被选为重要特征的次数除以它所在的子集被测试的次数）

from sklearn.linear_model import RandomizedLasso  #随机Lasso
from sklearn.datasets import load_boston
boston = load_boston()
#using the Boston housing data.
#Data gets scaled automatically by sklearn's implementation
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]
rlasso = RandomizedLasso(alpha=0.025) #alpha自动选择最优的值
rlasso.fit(X, Y)
print "Features sorted by their score:"      #得分：rlasso.scores_
print sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), names), reverse=True)

#结论：好的特征不会因为有相似的特征、关联特征而得分为0，这跟Lasso是不同的。
#对于特征选择任务，在许多数据集和环境下，稳定性选择往往是性能最好的方法之一


#4.2 递归特征消除 (Recursive feature elimination (RFE))    最优特征子集贪心算法
#反复的构建模型（如SVM或者回归模型）然后选出最好的（或者最差的）的特征（可以根据系数来选），把选出来的特征放到一遍，
#然后在剩余的特征上重复这个过程，直到所有特征都遍历了。这个过程中特征被消除的次序就是特征的排序

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression,Ridge
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]
#use linear regression as the model
lr = LinearRegression()
#lr=Ridge(alpha=5)
#rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(X,Y)
print "Features sorted by their rank:"    #得分：rfe.ranking_
print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))



