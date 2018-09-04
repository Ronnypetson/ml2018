import numpy as np

class modelo_linear:
	# Initializer
	def __init__(self,learning_rate=0.1,train_iter=1000,mini_batch_len=100):
		self.learning_rate = learning_rate
		self.train_iter = train_iter
		self.mini_batch_len = mini_batch_len

	# Compute mean squared loss
	def loss(Y,Y_):
		return np.mean((Y-Y_)**2)

	# Fit parameters theta by mini-batch gradient descent
	def fit(self,X_train,Y_train):
		# Create column with 1
		#X_train = np.insert(X_train,0,1,1)
		# Initialize parameters with small random values
		self.theta = np.random.normal(0.0,1.0,size=X_train[0].shape)
		mean_losses = np.zeros(self.train_iter)
		# Update parameters iteratively
		for i in range(self.train_iter):
			grad = np.zeros(self.theta.shape)
			indices = np.random.choice(X_train.shape[0],self.mini_batch_len,replace=False)
			# Compute gradient
			for ind in indices:
				x = X_train[ind]
				y = Y_train[ind]
				grad += (np.dot(x,self.theta)-y)*x
				mean_losses[i] += (np.dot(x,self.theta)-y)**2
			grad *= self.learning_rate/self.mini_batch_len
			mean_losses[i] /= self.mini_batch_len
			# Update parameters
			self.theta -= grad
		return mean_losses

	# Estimate Y from X
	def predict(self,X_test):
		#X_test = np.insert(X_test,0,1,1)
		return np.dot(X_test,self.theta)

