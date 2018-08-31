# Lembrar de botar a coluna com 1

import numpy as np
from preprocess import getXY
from grad_desc import *
from sklearn import linear_model
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

train_X,train_Y = getXY()
test_X, test_Y = getXY('diamonds-dataset/diamonds-test.csv')

#train_X = np.random.uniform(-1.0,1.0,size=(100,10))
#train_X = np.insert(train_X,0,1,1)
#theta = np.random.normal(1.0,5.0,size=train_X.shape[1])
#train_Y = np.dot(train_X,theta) + np.random.normal(0.0,0.1,size=100)

#plt.plot(train_X,train_Y,'ro')
#plt.show()

#test_X = np.random.uniform(-1.0,1.0,size=(100,10))
#test_X = np.insert(test_X,0,1,1)
#test_Y = np.dot(test_X,theta) + np.random.normal(0.0,0.1,size=100)

## (1) SGD regressor (sklearn)
#sk_regressor = linear_model.SGDRegressor()
#sk_regressor.fit(train_X,train_Y)
#print(sk_regressor.score(test_X,test_Y))

## (2) Descida de gradiente
mod_l = modelo_linear()
mse = mod_l.fit(train_X,train_Y)

#print(mse)
plt.plot(mse)
plt.show()

Y_ = mod_l.predict(test_X)
print(r2_score(test_Y,Y_))

## (3) Equacao normal
#X_T = np.transpose(train_X)
#X_T_X = np.dot(X_T,train_X)
#X_T_X_inv = np.linalg.inv(X_T_X)
#X_T_X_inv_X_T = np.dot(X_T_X_inv,X_T)
#theta = np.dot(X_T_X_inv_X_T,train_Y)

#Y_ = np.dot(test_X,theta)
#print(r2_score(test_Y,Y_))

