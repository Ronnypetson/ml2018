import numpy as np
from preprocess import getXY
from grad_desc import *
from sklearn import linear_model
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

def sgd_sklearn(train_X,train_Y,valid_X,valid_Y,learning_rate=0.1,max_iter=1000):
	## (1) SGD regressor (sklearn)
	sk_regressor = linear_model.SGDRegressor(max_iter=max_iter,eta0=learning_rate)
	sk_regressor.fit(train_X,train_Y)
	return r2_score(valid_Y,sk_regressor.predict(valid_X)), sk_regressor.coef_ # sk_regressor.score(valid_X,valid_Y)

def descida_gradiente(train_X,train_Y,valid_X,valid_Y,learning_rate=0.1):
	## (2) Descida de gradiente
	mod_l = modelo_linear(learning_rate=learning_rate)
	mse = mod_l.fit(train_X,train_Y)
	plt.xlabel('iterations')
	plt.ylabel('MSE')
	plt.plot(mse)
	plt.show()
	Y_ = mod_l.predict(valid_X)
	return r2_score(valid_Y,Y_), mod_l.theta

def eq_normal(train_X,train_Y,test_X,test_Y):
	## (3) Equacao normal
	X_T = np.transpose(train_X)
	X_T_X = np.dot(X_T,train_X)
	X_T_X_inv = np.linalg.inv(X_T_X)
	X_T_X_inv_X_T = np.dot(X_T_X_inv,X_T)
	theta = np.dot(X_T_X_inv_X_T,train_Y)
	Y_ = np.dot(test_X,theta)
	return r2_score(test_Y,Y_), theta

## Read data from csv files
# Load train-validation
train_X,train_Y,train_means,train_var = getXY('diamonds-dataset/diamonds-train.csv',train_norm=True)
# Load test
test_X,test_Y = getXY('diamonds-dataset/diamonds-test.csv',col_means=train_means,col_var=train_var)

# Aplicar validacaoo cruzada com k=5
num_samples = train_X.shape[0]
num_features = train_X.shape[1]
k = 5
block_len = int(num_samples/k)
methods = {"eq_normal":eq_normal,"descida_gradiente":descida_gradiente,"sgd_sklearn":sgd_sklearn}
for m in methods:
	print("Evaluating method "+m)
	r2_scores = np.zeros(k)
	params = np.zeros((k,num_features))
	for i in range(k):
		_valid_X = train_X[i*block_len:(i+1)*block_len]
		_train_X = np.concatenate((train_X[0:i*block_len],train_X[(i+1)*block_len:]),axis=0)
		_valid_Y = train_Y[i*block_len:(i+1)*block_len]
		_train_Y = np.concatenate((train_Y[0:i*block_len],train_Y[(i+1)*block_len:]),axis=0)
		r2_scores[i],params[i] = methods[m](_train_X,_train_Y,_valid_X,_valid_Y)
	print(r2_scores)
	print("Validation mean R2 score: %f"%np.mean(r2_scores))
	chosen_params = params[np.argmax(r2_scores)]
	print("Test R2 score: %f"%r2_score(test_Y,np.dot(test_X,chosen_params)))

