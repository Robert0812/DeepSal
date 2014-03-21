from sals.utils.DataHelper import DataMan_mnist
from sals.utils.FunctionHelper import *
from sals.models import sgd_optimizer

import numpy as np
import theano 
import theano.tensor as T 
import time 

class FCLayer(object):
	''' Fully-connected layer'''

	def __init__(self, n_in, n_out, input = None, 
				W_init = None, b_init = None, actfun=None, tag='') :

		print 'building model: Fully-connected layer' + tag 
		if input is not None:
			self.x = input 
		else:
			self.x = T.matrix('x')

		if W_init is None:

			wbound = np.sqrt(6./(n_in + n_out))

			if actfun is T.nnet.sigmoid: wbound *= 4

			rng = np.random.RandomState(1000)
			W_values =  np.asarray(rng.uniform(low = -wbound, high= wbound, 
				size=(n_in, n_out)), dtype = theano.config.floatX)							

			self.W = theano.shared(value = W_values, name = 'W'+tag, borrow = True)

		else:

			self.W = W_init

		if b_init is None:
			
			b_values = np.zeros((n_out,), dtype = theano.config.floatX)
			
			self.b = theano.shared(value = b_values, name = 'b'+tag, borrow = True)

		else:
			self.b = b_init

		self.actfun = actfun

		self.params = [self.W, self.b]

	def output(self):
		# feed forward output
		y = T.dot(self.x, self.W) + self.b

		if self.actfun is None:
			return y 
		else:
			return self.actfun(y)

	def regW(self, L):

		return self.W.norm(L)/np.prod(self.W.get_value().shape)


class GeneralModel(object):
	''' a wapper for general model '''
	def __init__(self, input, output, target, params,
					cost_func, error_func, regularizers=0):

		self.x = input
		self.ypred = output
		self.y = target
		self.params = params
		self.regularizers = regularizers
		self.cost_func = cost_func
		self.error_func = error_func

	def costs(self):

		return self.cost_func(self.ypred, self.y) + self.regularizers

	def errors(self):

		return self.error_func(self.ypred, self.y)

	def updates(self, lr):
		gparams = T.grad(cost = self.costs(), wrt = self.params)
		updates = [(self.params[p], self.params[p] - lr*gparams[p]) 
			for p in range(len(self.params))]
		return updates 

if __name__ == '__main__':

	mnist = DataMan_mnist('../data/mnist.pkl.gz')
	cpudata = mnist.load()
	mnist.share2gpumem(cpudata)

	x = T.matrix('x')
	y = T.ivector('y')
	
	test_id = 1

	if test_id == 1:

		fc0 = FCLayer(input=x, n_in=28*28, n_out=400, tag='_fc0')
		fc1 = FCLayer(input=fc0.output(), n_in=400, n_out=100, tag='_fc1')
		fc2 = FCLayer(input=fc1.output(), n_in=100, n_out=10, actfun=softmax, tag='_fc2')
		params_cmb = fc0.params + fc1.params + fc2.params
		#params_cmb = fc0.params + fc2.params
		ypred = fc2.output()

	elif test_id == 2:
		fc1 = FCLayer(input=x, n_in=28*28, n_out=10, actfun=T.nnet.softmax, tag='_fc1')
		params_cmb = fc1.params
		ypred = fc1.output()
	
	model = GeneralModel(input=x, output=ypred,
				target=y, params=params_cmb, 
				regularizers = 0,
				cost_func=mean_nll,
				error_func=mean_nneq)

	sgd = sgd_optimizer(data = mnist,  
					model = model,
					batch_size=1200, 
					learning_rate=0.1,
					n_epochs=200)
	sgd.fit()

