from sals.models import sgd_optimizer
from sals.models import FCLayer, GeneralModel, ConvLayer
from sals.utils.DataHelper import DataMan_msra
from sals.utils.FunctionHelper import *

import numpy as np
import theano 
import theano.tensor as T 
import time 

if __name__ == '__main__':

	msra = DataMan_msra('../data/msra_aug.pkl')
	cpudata = msra.load()
	msra.share2gpumem(cpudata)

	bs = 1000
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
	fc3 = FCLayer(input=conv2.output(), n_in=nfilter2*outL2*outL2, n_out=imL*imL, tag='_fc3')
	fc4 = FCLayer(input=fc3.output(), n_in=imL*imL, n_out=imL*imL, actfun=sigmoid, tag='_fc4')
	params_cmb = conv1.params + conv2.params + fc3.params + fc4.params
	#params_cmb = fc0.params + fc2.params
	#ypred = fc2.output().reshape((bs, imL, imL))
	ypred = fc4.output()

	model = GeneralModel(input=x, data = msra,
				output=ypred, target=y, 
				params=params_cmb, 
				cost_func=mean_nneq_cross_entropy,
				error_func=mean_sqr,
				regularizers = 0,
				batch_size=bs)

	sgd = sgd_optimizer(data = msra,  
					model = model,
					batch_size=bs, 
					valid_loss_delta = 0.01,
					learning_rate=0.001,
					learning_rate_decay=0.9,					
					momentum = 0,
					n_epochs=-1)
	sgd.fit()

	# evaluation and testing
	test_x = msra.test_x.get_value(borrow=True)
	test_y = msra.test_y.get_value(borrow=True)
	n_test = test_x.shape[0]
	n_batches_test = n_test/bs
	test_ypred = [model.test(i)[-1] for i in xrange(n_batches_test)]
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