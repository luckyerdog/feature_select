# -*- coding: utf-8 -*-


#2 线性模型和正则化

#2.1线性模型
from sklearn.linear_model import LinearRegression
import numpy as np

np.random.seed(0)
size = 5000
#A dataset with 3 features
X = np.random.normal(0, 1, (size, 3))
#Y = X0 + 2*X1 + noise
Y = X[:,0] + 2*X[:,1] + np.random.normal(0, 2, size)
lr = LinearRegression()
lr.fit(X, Y)
print lr.coef_,len(lr.coef_)

#A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)

print "Linear model:", pretty_print_linear(lr.coef_)



#若存在多个互相关联的特征（X1=X2=X3），这时候模型就不稳定
from sklearn.linear_model import LinearRegression
size = 100
np.random.seed(seed=5)
X_seed = np.random.normal(0, 1, size)
X1 = X_seed + np.random.normal(0, .1, size)
X2 = X_seed + np.random.normal(0, .1, size)
X3 = X_seed + np.random.normal(0, .1, size)
Y = X1 + X2 + X3 + np.random.normal(0,1, size)
X = np.array([X1, X2, X3]).T
lr = LinearRegression()
lr.fit(X,Y)
print "Linear model:", pretty_print_linear(lr.coef_)


#2.2 正则化 （当用在线性模型上时，L1正则化和L2正则化也称为Lasso和Ridge）

#L1正则化（有互相关联特征时不稳定）
from sklearn.linear_model import Lasso  #为线性回归提供了Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
boston = load_boston()
scaler = StandardScaler()  #StandardScaler能够把feature按照列转换成mean=0,standard deviation=1的正态分布
X = scaler.fit_transform(boston["data"])
Y = boston["target"]
names = boston["feature_names"]
lasso = Lasso(alpha=.3)  #alpha越大，模型越稀疏，越多特征系数变为0，
lasso.fit(X, Y)
print "Lasso model: ", pretty_print_linear(lasso.coef_, names, sort = True)

#L2正则化（稳定的模型，惩罚项中系数是二次项，系数取值变得平均，能力强的特征对应系数是非零)
from sklearn.linear_model import Ridge   #可用于互相关联特征
from sklearn.metrics import r2_score
size = 100
#We run the method 10 times with different random seeds
for i in range(10):
    print "Random seed %s" % i np.random.seed(seed=i)
    X_seed = np.random.normal(0, 1, size)
    X1 = X_seed + np.random.normal(0, .1, size)
    X2 = X_seed + np.random.normal(0, .1, size)
    X3 = X_seed + np.random.normal(0, .1, size)
    Y = X1 + X2 + X3 + np.random.normal(0, 1, size)
    X = np.array([X1, X2, X3]).T
    
    lr = LinearRegression()
    lr.fit(X,Y)
    print "Linear model:", pretty_print_linear(lr.coef_)
    
    ridge = Ridge(alpha=10)
    ridge.fit(X,Y)
    print "Ridge model:", pretty_print_linear(ridge.coef_) print    
    #结论：可以看出，不同的数据上线性回归得到的模型（系数）相差甚远，但对于L2正则化模型来说，
    #结果中的系数非常的稳定，差别较小，都比较接近于1，能够反映出数据的内在结构。




