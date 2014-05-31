import numpy as np
import theano
import theano.tensor as T 
from theano.tensor.signal.downsample import max_pool_2d 
from sals.utils.FunctionHelper import mean_nll, mean_nneq
import time

from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool

from theano.tensor.shared_randomstreams import RandomStreams

''' implement of dropout from https://github.com/mdenil/dropout/ '''
def drop_from_layer(rng, layer, p=0.5):
    srng = T.shared_randomstreams.RandomStreams(rng.randint(123))
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = layer*T.cast(mask, theano.config.floatX)
    return output
		
class FCLayer(object):
	''' Fully-connected layer'''

	def __init__(self, n_in, n_out, input = None, 
				W_init = None, b_init = None, 
				actfun=None, dropoutFrac=0, tag='') :

		print 'building model: Fully-connected layer{}, input:{}, output:{}'.format(
			tag, (np.nan, n_in), (np.nan, n_out)) 
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


	def __init__(self, image_shape, filter_shape, 
			input = None, W_init = None, b_init = None, 
			actfun=None, flatten = False, tag='') :

		outL = image_shape[-1]-filter_shape[-1]+1
		output_shape = (image_shape[0], filter_shape[0], outL, outL)
		print 'building model: Convolutional layer{}, input:{}, output:{}'.format(
			tag, image_shape, output_shape) 
		
		if input is not None:
			self.x = input 
		else:
			self.x = T.tensor4('x')

		fan_in = np.prod(filter_shape[1:])
		fan_out = filter_shape[0] * np.prod(filter_shape[2:])

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
		
		self.params = [self.W, self.b]

	def output(self):
		# convolution output
		# conv_out = T.nnet.conv.conv2d(
		# 			input=self.x, filters=self.W, 
		# 			filter_shape = self.filter_shape, 
		# 			image_shape=self.image_shape)
		
		input_shuffled = self.x.dimshuffle(1, 2, 3, 0) # bc01 to c01b
		filters_shuffled = self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
		conv_op = FilterActs(stride=1, partial_sum=1)
		contiguous_input = gpu_contiguous(input_shuffled)
		contiguous_filters = gpu_contiguous(filters_shuffled)
		conv_out_shuffled = conv_op(contiguous_input, contiguous_filters)

		y = conv_out_shuffled.dimshuffle(3, 0, 1, 2) + self.b.dimshuffle('x', 0, 'x', 'x')

		if self.actfun is not None: y = self.actfun(y)

		if self.flatten is True:
			y = y.flatten(2)

		return y

	def regW(self, L):

		return self.W.norm(L)/np.prod(self.W.get_value().shape)


class ConvPoolLayer(object):
	'''
	Convolutional + Pooling layer

	image_shape: (batch size, num input feature maps, image height, image width)

	filter_shape: (number of filters, num input feature maps, filter height,filter width)

	pool_shape: tuple or list of length 2

	'''


	def __init__(self, image_shape, filter_shape, pool_shape, 
			input = None, W_init = None, b_init = None, 
			actfun=None, flatten = False, tag='') :

		outL = np.floor((image_shape[-1]-filter_shape[-1]+1.)/pool_shape[-1]).astype(np.int)
		output_shape = (image_shape[0], filter_shape[0], outL, outL)
		print 'building model: Convolutional (Pooling) layer{}, input:{}, output:{}'.format(
			tag, image_shape, output_shape) 
		
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
		# conv_out = T.nnet.conv.conv2d(
		# 			input=self.x, filters=self.W, 
		# 			filter_shape = self.filter_shape, 
		# 			image_shape=self.image_shape)
		
		input_shuffled = self.x.dimshuffle(1, 2, 3, 0) # bc01 to c01b
		filters_shuffled = self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
		conv_op = FilterActs(stride=1, partial_sum=1)
		contiguous_input = gpu_contiguous(input_shuffled)
		contiguous_filters = gpu_contiguous(filters_shuffled)
		conv_out_shuffled = conv_op(contiguous_input, contiguous_filters)

		# max-pooling output
		# pooled_out = max_pool_2d(
		# 		input = conv_out,
		# 		ds = self.pool_shape,
		# 		ignore_border=True)

		pool_op = MaxPool(ds=self.pool_shape[0], stride=self.pool_shape[0])
		pooled_out_shuffled = pool_op(conv_out_shuffled)
		pooled_out = pooled_out_shuffled.dimshuffle(3, 0, 1, 2) # c01b to bc01

		y = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')

		if self.actfun is not None: y = self.actfun(y)

		if self.flatten is True:
			y = y.flatten(2)

		return y

	def regW(self, L):

		return self.W.norm(L)/np.prod(self.W.get_value().shape)


class GeneralModel(object):
	''' a wapper for general model '''
	def __init__(self, input, data, output, target, params,
					cost_func, error_func, regularizers=0, batch_size=100):

		self.x = input
		self.ypred = output
		self.y = target
		self.params = params
		self.regularizers = regularizers
		self.cost_func = cost_func
		self.error_func = error_func
		self.batch_size = batch_size	

		create_incs = lambda p: theano.shared(
            np.zeros_like(p.get_value(borrow=True)), borrow=True)

		self.incs = [create_incs(p) for p in self.params]

		index = T.lscalar()
		lr = T.fscalar()
		# indices = T.lvector()
		momentum = T.fscalar()
		batch_data = T.fmatrix()
		batch_label = T.fvector()
		
		n_train = data.train_x.shape[0]

		self.train = theano.function(inputs=[batch_data, batch_label, lr, momentum], 
			outputs=[self.costs(), self.errors(), self.outputs()], 
			updates=self.updates(lr, momentum),
			givens={
				self.x: batch_data,
				self.y: batch_label
			})

		self.test = theano.function(inputs=[batch_data, batch_label],
			outputs = [self.errors(), self.outputs()], 
			givens = {
				self.x : batch_data,
				self.y : batch_label
		})

		self.valid = theano.function(inputs=[batch_data, batch_label], 
			outputs=self.errors(), 
			givens={
				self.x: batch_data, 
				self.y: batch_label
			})

		# self.train = theano.function(inputs=[indices, lr, momentum], 
		# 	outputs=[self.costs(), self.errors(), self.outputs()], 
		# 	updates=self.updates(lr, momentum),
		# 	givens={
		# 		self.x: data.train_x[indices],
		# 		self.y: data.train_y[indices]
		# 	})

		# self.train = theano.function(inputs=[index, lr, momentum], 
		# 	outputs=[self.costs(), self.errors(), self.outputs()], 
		# 	updates=self.updates(lr, momentum),
		# 	givens={
		# 		self.x: data.train_x[index*batch_size : (index+1)*batch_size],
		# 		self.y: data.train_y[index*batch_size : (index+1)*batch_size]
		# 	})

		# self.test = theano.function(inputs=[index,],
		# 	outputs = [self.errors(), self.outputs()], 
		# 	givens = {
		# 		self.x : data.test_x[index*batch_size:(index+1)*batch_size],
		# 		self.y : data.test_y[index*batch_size:(index+1)*batch_size]
		# })

		# self.valid = theano.function(inputs=[index,], 
		# 	outputs=self.errors(), 
		# 	givens={
		# 		self.x: data.valid_x[index*batch_size : (index+1)*batch_size],
		# 		self.y: data.valid_y[index*batch_size : (index+1)*batch_size]
		# 	})

	def costs(self):

		return self.cost_func(self.ypred, self.y) + self.regularizers

	def errors(self):

		return self.error_func(self.ypred, self.y)

	def updates(self, lr, momentum):
		gparams = T.grad(cost = self.costs(), wrt = self.params)

		updates_incs = [(self.incs[p], momentum*self.incs[p] - lr*gparams[p]) 
				for p in range(len(self.params))]

		updates = [(self.params[p], self.params[p] + momentum*self.incs[p] - lr*gparams[p]) 
			for p in range(len(self.params))]
		return updates

	def outputs(self):
		return self.ypred + self.y*0


class sgd_optimizer(object):
	'''
	stochastic gradient descent optimization
	'''
	def __init__(self, data, model, batch_size=100, 
		learning_rate=0.1,
		valid_loss_delta = 1e-2,
		learning_rate_decay=0.95,
		momentum = 0.9,
		n_epochs=200):

		self.data = data 
		self.batch_size = batch_size
		if n_epochs > 0:
			self.n_epochs = n_epochs
		else:
			self.n_epochs = np.inf

		self.model = model
		self.lr = learning_rate
		self.lr_decay = learning_rate_decay
		self.valid_loss_delta = valid_loss_delta
		self.momentum = momentum

	def fit(self):

		print 'fitting ...'
		n_batches_train = self.data.train_x.get_value(borrow=True).shape[0]/self.batch_size
		n_batches_valid = self.data.valid_x.get_value(borrow=True).shape[0]/self.batch_size
		n_batches_test = self.data.test_x.get_value(borrow=True).shape[0]/self.batch_size
		index_show = np.floor(np.linspace(0, n_batches_train-1, 5))

		start_time = time.clock()
		epoch = 0
		valid_loss_prev = 2304
		check_period = 10.
		count = 0
		while (epoch < self.n_epochs):
			epoch += 1
			#print self.model.params[0].get_value().max()
			for batch_index in range(n_batches_train):
				t0 = time.clock()
				batch_avg_cost, batch_avg_error, _ = self.model.train(batch_index, epoch, self.lr, self.momentum)
				t1 = time.clock()
				if batch_index in index_show:
					print '{0:d}.{1:02d}... cost: {2:.3f}, error: {3:.3f} ({4:.3f} sec)'.format(epoch,
						batch_index, batch_avg_cost*100/2304, batch_avg_error*100/2304, t1-t0)

			valid_avg_loss = np.mean([self.model.valid(i) for i in range(n_batches_valid)])
			test_avg_loss = np.mean([self.model.test(i)[0] for i in xrange(n_batches_test)])
			
			if valid_avg_loss/2304. < 10/100.:
				# decrease = (valid_loss_prev - valid_avg_loss)/valid_loss_prev
				# if decrease > self.valid_loss_delta:
				count += 1
				delta_loss = valid_loss_prev - valid_avg_loss
				if count == check_period and delta_loss/2304. < 0.01/100.:
					self.lr *= self.lr_decay
					valid_loss_prev = valid_avg_loss
					count = 0
			print '==================Test Output==================='
			print 'Update learning_rate {0:.6f}'.format(self.lr) if count == 0 else 'no update'
			print 'validation error {0:.3f}%, testing error {1:.3f}%'.format(  
				valid_avg_loss*100./2304, test_avg_loss*100./2304)
			print '================================================'

		end_time = time.clock()
		print 'The code run for %d epochs, with %f epochs/sec' % (
        			epoch, 1. * epoch / (end_time - start_time))

	def fit_viper(self):

		print 'fitting ...'
		n_train = self.data.train_x.shape[0]
		n_batches_train = np.int(self.data.train_x.shape[0]/(self.batch_size*1.0))
		n_batches_valid = np.int(self.data.valid_x.shape[0]/(self.batch_size*1.0))
		n_batches_test = np.int(self.data.test_x.shape[0]/(self.batch_size*1.0))
		index_show = np.floor(np.linspace(0, n_batches_train-1, 10))

		start_time = time.clock()
		epoch = 0
		valid_loss_prev = 0.5
		check_period = 10.
		count = 0
		train_error = np.zeros(n_batches_train)

		self.data.train_x = self.data.train_x.astype(np.float32)
		self.data.train_y = self.data.train_y.astype(np.float32)
		self.data.valid_x = self.data.valid_x.astype(np.float32)
		self.data.valid_y = self.data.valid_y.astype(np.float32)
		self.data.test_x = self.data.test_x.astype(np.float32)
		self.data.test_y = self.data.test_y.astype(np.float32)

		while (epoch < self.n_epochs):
			epoch += 1
			# generate a random set of batch indices
			np.random.seed(epoch)
			randidx = np.random.permutation(n_train)

			for batch_index in range(n_batches_train):

				this_batch_indices = randidx[batch_index*self.batch_size : (batch_index+1)*self.batch_size]

				t0 = time.clock()
				batch_avg_cost, batch_avg_error, _ = self.model.train(self.data.train_x[this_batch_indices],
															self.data.train_y[this_batch_indices], 
															self.lr, self.momentum)
				t1 = time.clock()

				train_error[batch_index] = batch_avg_error

				if batch_index in index_show:
					print '({0:d}.{1:d}): {2:03d}.{3:03d}... cost: {4:.6f}, error: {5:.6f} ({6:.3f} sec)'.format(self.n_epochs, n_batches_train,
						epoch, batch_index+1, float(batch_avg_cost), float(batch_avg_error), float(t1-t0))

			train_avg_loss = train_error.mean()

			valid_avg_loss = np.mean([self.model.valid(self.data.valid_x[i*self.batch_size:(i+1)*self.batch_size], 
				self.data.valid_y[i*self.batch_size:(i+1)*self.batch_size]) for i in range(n_batches_valid)])
			
			test_avg_loss = np.mean([self.model.test(self.data.test_x[i*self.batch_size:(i+1)*self.batch_size], 
				self.data.test_y[i*self.batch_size:(i+1)*self.batch_size])[0] for i in range(n_batches_test)])
			
			if valid_avg_loss < 10/100.:
				# decrease = (valid_loss_prev - valid_avg_loss)/valid_loss_prev
				# if decrease > self.valid_loss_delta:
				# count += 1
				# delta_loss = valid_loss_prev - valid_avg_loss
				# if count == check_period and delta_loss < 0.01/100.:
				# 	self.lr *= self.lr_decay
				# 	valid_loss_prev = valid_avg_loss
				# 	count = 0
				count += 1
				if count == check_period:
					self.lr *= self.lr_decay
					count = 0

			print '===========================Test Output==========================='
			print 'Update learning_rate {0:.6f}'.format(self.lr) if count == 0 else 'no update'
			print 'train set error {0:.6f}, valid set error {1:.6f}'.format(train_avg_loss, valid_avg_loss)
			print 'test set error {0:.6f}'.format(test_avg_loss)
			print '================================================================='
			time.sleep(1)

		end_time = time.clock()
		print 'The code run for %d epochs, with %f epochs/sec' % (
        			epoch, 1. * epoch / (end_time - start_time))
