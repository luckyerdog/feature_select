# -*- coding: utf-8 -*-

#3 随机森林
    
#3.1 平均不纯度减少 (mean decrease impurity)
#对于一个决策树森林来说，可以算出每个特征平均减少了多少不纯度，并把它平均减少的不纯度作为特征选择的值
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#Load boston housing dataset as an example
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]
rf = RandomForestRegressor()
rf.fit(X, Y)
#model的feature_importances_ 参数排序
print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True)

#如下例，若特征关联，则先选择的特征，先重要性，后边选择的就不重要了(按理应该是一样)。特征随机选择方法ke稍微缓解这个问题
size = 10000
np.random.seed(seed=10)
X_seed = np.random.normal(0, 1, size)
X0 = X_seed + np.random.normal(0, .1, size)
X1 = X_seed + np.random.normal(0, .1, size)
X2 = X_seed + np.random.normal(0, .1, size)
X = np.array([X0, X1, X2]).T
Y = X0 + X1 + X2
rf = RandomForestRegressor(n_estimators=20, max_features=2) #20棵树
rf.fit(X, Y)
print "Scores for X0, X1, X2:", map(lambda x:round (x,3), rf.feature_importances_)
#关联特征的打分存在不稳定的现象，这不仅仅是随机森林特有的，大多数基于模型的特征选择方法都存在这个问题


#3.2 平均精确率减少 (Mean decrease accuracy)
#主要思路是打乱每个特征的特征值顺序，度量顺序变动对模型的精确率的影响。
#不重要的变量来说，打乱顺序对模型的精确率影响不会太大，但是对于重要的变量来说，打乱顺序就会降低模型的精确率。
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
X = boston["data"]
Y = boston["target"]
rf = RandomForestRegressor()
scores = defaultdict(list)
#crossvalidate the scores on a number of different random splits of the data
for train_idx, test_idx in ShuffleSplit(len(X), 100, .3): #分100次，比例为7:3，ShuffleSplit()返回选中的序号的列表
    #print train_idx
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    r = rf.fit(X_train, Y_train)
    acc = r2_score(Y_test, rf.predict(X_test)) #为打乱之前的得分
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, rf.predict(X_t))  #打乱某个特征顺序后的得分
        scores[names[i]].append((acc-shuff_acc)/acc)
print "Features sorted by their score:"
print sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True)







