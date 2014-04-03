from sals.models import sgd_optimizer
from sals.models import FCLayer, GeneralModel, ConvLayer
from sals.utils.DataHelper import DataMan_msra
from sals.utils.FunctionHelper import *

import numpy as np
import theano 
import theano.tensor as T 
import time 

if __name__ == '__main__':

	msra = DataMan_msra('../data/msra_norm3.pkl')
	cpudata = msra.load()
	msra.share2gpumem(cpudata)

	bs = 200
	imL = 48
	filterL = 5
	recfield = 2
	nfilter1 = 32
	nfilter2 = 32
	filterL2= 5

	x = T.matrix(name='x', dtype=theano.config.floatX)
	y = T.matrix(name='y', dtype=theano.config.floatX)
	
	layer0 = x.reshape((bs, 3, imL, imL))
	conv1 = ConvLayer(input = layer0, image_shape = (bs, 3, imL, imL),
			filter_shape =(nfilter1, 3, filterL, filterL),
			pool_shape = (recfield, recfield), 
			flatten = False, 
			actfun=tanh, 
			tag='_conv1')

	outL1 = np.floor((imL-filterL+1.)/recfield).astype(np.int)
	conv2 = ConvLayer(input = conv1.output(), image_shape=(bs, nfilter1, outL1, outL1),
			filter_shape = (nfilter2, nfilter1, filterL2, filterL2),
			pool_shape=(recfield, recfield),
			flatten=True,
			actfun=tanh,
			tag='_conv2')
	
	outL2 = np.floor((outL1-filterL2+1.)/recfield).astype(np.int)
	fc3 = FCLayer(input=conv2.output(), n_in=nfilter2*outL2*outL2, n_out=imL*imL, actfun=sigmoid, tag='_fc3')
	params_cmb = conv1.params + conv2.params + fc3.params 
	#params_cmb = fc0.params + fc2.params
	#ypred = fc2.output().reshape((bs, imL, imL))
	ypred = fc3.output()

	model = GeneralModel(input=x, output=ypred,
				target=y, params=params_cmb, 
				regularizers = 0,
				cost_func=mean_cross_entropy,
				error_func=mean_sqr)

	sgd = sgd_optimizer(data = msra,  
					model = model,
					batch_size=bs, 
					learning_rate=0.001,
					valid_loss_decay = 0.005,
					learning_rate_decay=1,
					n_epochs=1000)
	sgd.fit()

	# evaluation and testing
	train, valid, test = cpudata
	test_x, test_y = test 

	index = T.lscalar()
	test_model = theano.function(inputs=[index,],
		outputs = model.outputs(), 
		givens = {
			model.x : msra.test_x[index*bs:(index+1)*bs],
			model.y : msra.test_y[index*bs:(index+1)*bs]
		})

	n_test = test_x.shape[0]
	n_batches_test = n_test/bs
	test_ypred = [test_model(i) for i in xrange(n_batches_test)]
	test_ypred = np.asarray(test_ypred).reshape(n_test, -1)

	T = 20
	thrs = np.linspace(1, 0, T)
	fbeta = np.zeros(T)

	for i in range(len(thrs)):
		test_ypred_binary = map(lambda x: x>=thrs[i], test_ypred)
		fbeta[i] = np.mean(map(lambda x, y: get_fbeta(x, y), test_y, test_ypred_binary))
	
	#pl.plot(rocs[0, :], rocs[1, :])
	print fbeta.sum()
	import pylab as pl
	pl.figure()
	pl.subplot(1, 2, 1) 
	pl.imshow(test_y[0, :].reshape(imL, imL))
	pl.subplot(1, 2, 2)
	pl.imshow(test_ypred[0, :].reshape(imL, imL))
	pl.show()