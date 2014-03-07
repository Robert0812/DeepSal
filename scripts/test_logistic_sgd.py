from sals.utils.Data import DataMan_mnist

import numpy as np
import theano
import theano.tensor as T 

class LogisticRegression(object):
	''' 
		define the learning cost, evaluation error, and update 
	'''

	def __init__(self, dim, n_class):

		print 'building model: logistic regression'
		self.x = T.matrix('x')
		self.y = T.ivector('y')

		self.W = theano.shared(value=np.zeros((dim, n_class), 
			dtype = theano.config.floatX), 
			name= 'W', borrow=True)

		self.b = theano.shared(value=np.zeros((n_class,), 
			dtype = theano.config.floatX), 
			name = 'b', borrow=True)

		self.p_y_given_x = T.nnet.softmax(T.dot(self.x, self.W) + self.b)

		self.y_pred = T.argmax(self.p_y_given_x, axis=1)

		self.params = [self.W, self.b]


	def negative_log_likelihood(self):

		return -T.mean(T.log(self.p_y_given_x)[T.arange(self.y.shape[0]), self.y])


	def costs(self):

		return self.negative_log_likelihood()


	def errors(self):

		if self.y.ndim != self.y_pred.ndim:
			raise ValueError('y should have the same shape as self.y_pred')

		if self.y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred, self.y))
		else:
			raise NotImplementedError()


	def updates(self, learning_rate):
		'''
			return update rules
		'''
		g_W = T.grad(cost=self.negative_log_likelihood(), wrt=self.W)
		g_b = T.grad(cost=self.negative_log_likelihood(), wrt=self.b)
		update_w = (self.W, self.W - learning_rate * g_W)
		update_b = (self.b, self.b - learning_rate * g_b)
		updates = [update_w, update_b]
		return updates

    	
class sgd_optimizer(object):

	def __init__(self, data, model, batch_size, learning_rate, n_epochs):

		self.data = data 
		self.batch_size = batch_size
		self.lr = learning_rate
		self.n_epochs = n_epochs
		self.model = model

	def fit(self):

		index = T.lscalar()
		test_model = theano.function(inputs=[index,], 
			outputs=self.model.errors(), 
			givens={
				self.model.x: self.data.test_x[index*self.batch_size : (index+1)*self.batch_size],
				self.model.y: self.data.test_y[index*self.batch_size : (index+1)*self.batch_size]
			})

		valid_model = theano.function(inputs=[index,], 
			outputs=self.model.errors(), 
			givens={
				self.model.x: self.data.valid_x[index*self.batch_size : (index+1)*self.batch_size],
				self.model.y: self.data.valid_y[index*self.batch_size : (index+1)*self.batch_size]
			})

		train_model = theano.function(inputs=[index,], 
			outputs=self.model.costs(), 
			updates=self.model.updates(self.lr),
			givens={
				self.model.x: self.data.train_x[index*self.batch_size : (index+1)*self.batch_size],
				self.model.y: self.data.train_y[index*self.batch_size : (index+1)*self.batch_size]
			})

		print 'fitting ...'
		n_batches_train = self.data.train_x.get_value(borrow=True).shape[0]/self.batch_size
		n_batches_valid = self.data.valid_x.get_value(borrow=True).shape[0]/self.batch_size
		n_batches_test = self.data.test_x.get_value(borrow=True).shape[0]/self.batch_size

		epoch = 0
		done_looping = False 
		
		while (epoch < self.n_epochs) and (not done_looping):
			epoch += 1

			for batch_index in range(n_batches_train):
				batch_avg_cost = train_model(batch_index)
				
				if epoch % 5 == 0:
					valid_losses = [valid_model(i) for i in range(n_batches_valid)]
					print 'epoch {0:03d}, minibatch {1:02d}/{2:02d}, validation error {3:.2f} %'.format(epoch, 
						batch_index, n_batches_train, np.mean(valid_losses)*100.)


if __name__ == '__main__':

	mnist = DataMan_mnist('../data/mnist.pkl.gz')
	cpudata = mnist.load()
	mnist.share2gpumem(cpudata)

	logreg = LogisticRegression(dim=28*28, n_class=10)

	sgd = sgd_optimizer(data = mnist,  
					model = logreg,
					batch_size=600, 
					learning_rate=0.13,
					n_epochs=200)
	sgd.fit()
