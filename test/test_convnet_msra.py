from sals.models import sgd_optimizer
from sals.models import FCLayer, GeneralModel, ConvLayer
from sals.utils.DataHelper import DataMan_msra
from sals.utils.FunctionHelper import *

import numpy as np
import theano 
import theano.tensor as T 
import time 

if __name__ == '__main__':

	msra = DataMan_msra('../data/msra.pkl')
	cpudata = msra.load()
	msra.share2gpumem(cpudata)

	bs = 200
	imL = 48
	filterL = 6
	recfield = 2
	nfilter1 = 32

	x = T.matrix('x')
	y = T.matrix('y')
	
	layer0 = x.reshape((bs, 3, imL, imL))
	conv1 = ConvLayer(input = layer0, image_shape = (bs, 3, imL, imL),
			filter_shape =(nfilter1, 3, filterL, filterL),
			pool_shape = (recfield, recfield), 
			flatten = True, 
			actfun=sigmoid)

	outL = (imL-filterL+1.)/recfield
	fc2 = FCLayer(input=conv1.output(), n_in=nfilter1*outL*outL*3, n_out=imL*imL, actfun=softmax, tag='_fc2')
	params_cmb = conv1.params + fc2.params 
	#params_cmb = fc0.params + fc2.params
	ypred = fc2.output().reshape((bs, imL, imL))

	model = GeneralModel(input=x, output=ypred,
				target=y, params=params_cmb, 
				regularizers = 0,
				cost_func=mean_cross_entropy,
				error_func=mean_cross_entropy)

	sgd = sgd_optimizer(data = msra,  
					model = model,
					batch_size=bs, 
					learning_rate=0.1,
					n_epochs=200)
	sgd.fit()
