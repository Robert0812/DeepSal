from sals.utils.DataHelper import DataMan_mnist
from sals.models import LogisticRegression

import numpy as np
import theano
import theano.tensor as T 
import time 
    	
class sgd_optimizer(object):

	def __init__(self, data, model, batch_size, learning_rate, n_epochs):

		self.data = data 
		self.batch_size = batch_size
		self.lr = learning_rate
		self.n_epochs = n_epochs
		self.model = model

	def run(self):

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

		start_time = time.clock()
		epoch = 0
		while (epoch < self.n_epochs):
			epoch += 1
			for batch_index in range(n_batches_train):
				batch_avg_cost = train_model(batch_index)
				
				if epoch % 5 == 0:
					valid_losses = [valid_model(i) for i in range(n_batches_valid)]
					test_losses = [test_model(i) for i in xrange(n_batches_test)]
					print 'epoch {0:03d}, minibatch {1:02d}/{2:02d}, validation error {3:.2f} %, testing error {4:.2f} %'.format(epoch, 
						batch_index, n_batches_train, np.mean(valid_losses)*100., np.mean(test_losses)*100.)

		end_time = time.clock()
		print 'The code run for %d epochs, with %f epochs/sec' % (
        			epoch, 1. * epoch / (end_time - start_time))
		print 'Final model:'
		print self.model.W.get_value(), self.model.b.get_value() 

if __name__ == '__main__':

	mnist = DataMan_mnist('../data/mnist.pkl.gz')
	cpudata = mnist.load()
	mnist.share2gpumem(cpudata)

	logreg = LogisticRegression(n_in=28*28, n_out=10)

	sgd = sgd_optimizer(data = mnist,  
					model = logreg,
					batch_size=600, 
					learning_rate=0.2,
					n_epochs=200)
	sgd.run()

