from sals.models import sgd_optimizer
from sals.models import FCLayer, GeneralModel, ConvLayer
from sals.utils.DataHelper import DataMan_msra
from sals.utils.FunctionHelper import *

import numpy as np
import theano 
import theano.tensor as T 
import time 

if __name__ == '__main__':

	msra = DataMan_msra('../data/msra_norm.pkl')
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

	test_model = theano.function(inputs=[],
		outputs = model.outputs(), 
		givens = {
			model.x : msra.test_x,
			model.y : msra.test_y
		})

	test_ypred = test_model()
	T = 20
	thrs = np.linspace(1, 0, T)
	rocs = np.zeros((2, T))

	for i in range(len(thrs)):
		test_ypred_binary = map(lambda x: x>=thrs[i], test_ypred)
		roc_pair = map(lambda x, y: getoc(x, y), test_y, test_ypred_binary)
		rocs[:, i] = np.asarray(roc_pair).mean(axis=0)
	
	#pl.plot(rocs[0, :], rocs[1, :])
	print thrs, rocs 