# Lembrar de botar a coluna com 1

from preprocess import getXY
from grad_desc import *
from sklearn import linear_model
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

train_X,train_Y = getXY()
test_X, test_Y = getXY('diamonds-dataset/diamonds-test.csv')
#sk_regressor = linear_model.SGDRegressor()
#sk_regressor.fit(train_X,train_Y)
#print(sk_regressor.score(test_X,test_Y))

mod_l = modelo_linear()
mse = mod_l.fit(train_X,train_Y)

print(mse)
plt.plot(mse)
plt.show()

Y_ = mod_l.predict(test_X)
print(r2_score(test_Y,Y_))

