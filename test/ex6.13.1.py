import theano
import numpy as np
from theano import function
from theano import shared 
import theano.tensor as T
from theano.tensor.nnet import sigmoid 

class NNet(object):

	def __init__(self, 
		input = T.dvector('input'), 
		target = T.dvector('target'), 
		n_input=1, n_hidden=1, n_output=1, 
		lr=1e-3, **kw):

		super(NNet, self).__init__(**kw)

		self.input = input 
		self.target = target 
		self.lr = shared(lr, 'learning_rate')
		self.w1 = shared(np.zeros((n_hidden, n_input)), 'w1')
		self.w2 = shared(np.zeros((n_output, n_hidden)), 'w2')

		self.hidden = sigmoid(T.dot(self.w1, self.input))
		self.output = T.dot(self.w2, self.hidden)
		self.cost = T.sum((self.output - self.target)**2) 

		self.sgd_updates = {
				self.w1: self.w1 - self.lr * T.grad(self.cost, self.w1), 
				self.w2: self.w2 - self.lr * T.grad(self.cost, self.w2)}

		self.sgd_step = function(inputs = [self.input, self.target], 
								outputs = [self.output, self.cost], 
								updates = self.sgd_updates)

		self.compute_output = function(inputs=[self.input], outputs=self.output)

		self.output_from_hidden = function(inputs = [self.hidden], outputs=self.output)
