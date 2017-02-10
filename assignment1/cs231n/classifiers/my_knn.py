import numpy as np

class myknn(object):
	def __int__(self):
		pass

	def train(self, X, Y):
		self.X_train = X
		self.Y_train = Y

	def predict(self, X, k=1, l=2):
		dists = self.compute_distances(X, l)
		if l == 1:
			dists = self.compute_L1_distances(X)
		elif l == 2:
			dists = self.compute_L2_distances(X)
		return self.predict_labels(dists, k)

	def compute_L1_distances(X):
		num_test = X.shape[0]
		dists = np.zeros(num_test)
		for i in xrange(num_test):
			dists[i] = np.sum(np.abs(self.X_train - X[i, :]), axis=1)


	def compute_L2_distances(self, X):
		num_test = X.shape[0]
		dists = np.zeros(num_test)
		-2*np.dot(X, self.X_train.T) + np.sum(np.square(self.X_train), axis = 1)
		for i in xrange(num_test):
			dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i, :])), axis=1)   
			#axis=1 compute sum of each row

	# def compute_L2_distances(self, X):
	# 	num_test = X.shape[0]
	# 	dists = np.zeros(num_test)
	# 	for i in xrange(num_test):
	# 		dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i, :])), axis=1)   
	# 		#axis=1 compute sum of each row

	def compute_L2_distances(self, X):
		num_test = X.shape[0]
		dists = np.zeros(num_test)
		dists = np.sqrt(-2*np.dot(X, self.X_train.T) + np.sum(np.square(X)) + np.sum(np.square(self.X_train), axis=1)) 


	def predict_labels(self, dists, k=1):
		num_test = dists.shape[0]
		y_pred = np.zeros(num_test)
		for i in xrange(num_test):
			#距离最小k个的类别
			closest_y = self.Y_train[np.argsort(dists[i])[:k]]
			#argsort函数返回的是数组值从小到大的索引值
			y_pred[i] = np.argmax(np.bincount(closest_y))
			#np.bincount() 第i个元素的值表示整数i在参数数组中出现的次数
		return y_pred

