from sals.utils.DataHelper import DataMan_mnist
from sals.utils.FunctionHelper import *
from sals.models import sgd_optimizer
from sals.models import FCLayer

import numpy as np
import theano 
import theano.tensor as T 
import time 

class ConvLayer(object):
	'''
	Convolutional layer

	image_shape: (batch size, num input feature maps, image height, image width)

	filter_shape: (number of filters, num input feature maps, filter height,filter width)

	pool_shape: tuple or list of length 2

	'''


	def __init__(self, image_shape, filter_shape, pool_shape, 
			input = None, W_init = None, b_init = None, 
			actfun=None, flatten = False, tag='') :

		print 'building model: convolutional layer' + tag 
		if input is not None:
			self.x = input 
		else:
			self.x = T.tensor4('x')

		fan_in = np.prod(filter_shape[1:])
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:])/np.prod(pool_shape))

		if W_init is None:

			wbound = np.sqrt(6./(fan_in + fan_out))

			if actfun is T.nnet.sigmoid: wbound *= 4

			rng = np.random.RandomState(1000)
			W_values =  np.asarray(rng.uniform(low = -wbound, high= wbound, 
				size=filter_shape), dtype = theano.config.floatX)							

			self.W = theano.shared(value = W_values, name = 'W'+tag, borrow = True)

		else:

			self.W = W_init

		if b_init is None:
			
			b_values = np.zeros((filter_shape[0],), dtype = theano.config.floatX)
			
			self.b = theano.shared(value = b_values, name = 'b'+tag, borrow = True)

		else:
			self.b = b_init

		self.actfun = actfun
		self.flatten  = flatten
		self.filter_shape = filter_shape
		self.image_shape = image_shape
		self.pool_shape = pool_shape

		self.params = [self.W, self.b]

	def output(self):
		# convolution output
		conv_out = T.nnet.conv.conv2d(
					input=self.x, filters=self.W, 
					filter_shape = self.filter_shape, 
					image_shape=self.image_shape)

		# max-pooling output
		pooled_out = T.signal.downsample.max_pool_2d(
				input = conv_out,
				ds = self.pool_shape,
				ignore_border=True)

		y = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')

		if self.actfun is not None: y = self.actfun(y)

		if self.flatten is True:
			y = y.flatten(2)

		return y

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

	bs = 600
	imL = 28
	filterL = 5
	recfield = 2
	nfilter1 = 32

	x = T.matrix('x')
	y = T.ivector('y')
	
	layer0 = x.reshape((bs, 1, imL, imL))
	conv1 = ConvLayer(input = layer0, image_shape = (bs, 1, imL, imL),
			filter_shape =(nfilter1, 1, filterL, filterL),
			pool_shape = (recfield, recfield), 
			flatten = True, 
			actfun=sigmoid)

	outL = (imL-filterL+1.)/recfield
	fc2 = FCLayer(input=conv1.output(), n_in=nfilter1*outL*outL, n_out=10, actfun=softmax, tag='_fc2')
	params_cmb = conv1.params + fc2.params 
	#params_cmb = fc0.params + fc2.params
	ypred = fc2.output()
	
	model = GeneralModel(input=x, output=ypred,
				target=y, params=params_cmb, 
				regularizers = 0,
				cost_func=mean_nll,
				error_func=mean_nneq)

	sgd = sgd_optimizer(data = mnist,  
					model = model,
					batch_size=bs, 
					learning_rate=0.1,
					n_epochs=200)
	sgd.fit()

