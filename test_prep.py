from preprocess import getXY
from sklearn import linear_model

train_X,train_Y = getXY()
test_X, test_Y = getXY('diamonds-dataset/diamonds-test.csv')
sk_regressor = linear_model.SGDRegressor()
sk_regressor.fit(train_X,train_Y)
print(sk_regressor.score(test_X,test_Y))

