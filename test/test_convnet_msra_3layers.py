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
				cost_func=mean_sqr_map,
				error_func=mean_sqr_map)

	sgd = sgd_optimizer(data = msra,  
					model = model,
					batch_size=bs, 
					learning_rate=0.001,
					valid_loss_decay = 0.005,
					learning_rate_decay=0.99,
					n_epochs=1000)
	sgd.fit()
