import numpy as np
import theano
import theano.tensor as T 
from sals.utils.FunctionHelper import mean_nll, mean_nneq
import time

class LogisticRegression(object):
	''' 
		define the learning cost, evaluation error, and update 
	'''

	def __init__(self, n_in, n_out,
	 			input=None, target=None, 
				actfunc=T.nnet.softmax, 
				costfunc=mean_nll, 
				errorfunc=mean_nneq):

		if input is not None:
			self.x = input 
		else:
			self.x = T.matrix('x')

		if target is not None:
			self.y = target
		else:
			self.y = T.ivector('y')

		self.actfunc = actfunc
		self.costfunc = costfunc
		self.errorfunc = errorfunc

		self.W = theano.shared(value=np.zeros((n_in, n_out), 
			dtype = theano.config.floatX), 
			name= 'W', borrow=True)

		self.b = theano.shared(value=np.zeros((n_out,), 
			dtype = theano.config.floatX), 
			name = 'b', borrow=True)

		self.output = self.actfunc(T.dot(self.x, self.W) + self.b)

		self.params = [self.W, self.b]

	def costs(self):
		return self.costfunc(self.output, self.y)
	
	def errors(self):

		if self.y.dtype.startswith('int'):
			return self.errorfunc(self.output, self.y)
		else:
			raise NotImplementedError()

	def updates(self, learning_rate):
		'''
			return update rules
		'''
		g_W = T.grad(cost=self.costs(), wrt=self.W)
		g_b = T.grad(cost=self.costs(), wrt=self.b)
		update_w = (self.W, self.W - learning_rate * g_W)
		update_b = (self.b, self.b - learning_rate * g_b)
		updates = [update_w, update_b]
		return updates

		
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

		start_time = time.clock()
		epoch = 0
		while (epoch < self.n_epochs):
			epoch += 1
			#print self.model.params[0].get_value().max()
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
		#print 'Final model:'
		#print self.model.W.get_value(), self.model.b.get_value() 