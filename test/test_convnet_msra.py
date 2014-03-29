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
	train, valid, test = cpudata
	train_x, train_y = train
	valid_x, valid_y = valid
	test_x, test_y = test
	train_x = np.asarray(train_x, dtype = np.float32)
	train_y = np.asarray(train_y, dtype =np.float32)
	valid_x = np.asarray(valid_x, dtype = np.float32)
	valid_y = np.asarray(valid_y, dtype = np.float32)
	test_x = np.asarray(test_x, dtype=np.float32)
	test_y = np.asarray(test_y, dtype = np.float32)
	train = [train_x, train_y]
	valid = [valid_x, valid_y]
	test = [test_x, test_y]
	cpudata_new = [train, valid, test]

	msra.share2gpumem(cpudata_new)

	bs = 200
	imL = 48
	filterL = 6
	recfield = 2
	nfilter1 = 32

	x = T.matrix('x')
	y = T.dtensor3('y')
	
	layer0 = x.reshape((bs, 3, imL, imL))
	conv1 = ConvLayer(input = layer0, image_shape = (bs, 3, imL, imL),
			filter_shape =(nfilter1, 3, filterL, filterL),
			pool_shape = (recfield, recfield), 
			flatten = True, 
			actfun=sigmoid)

	outL = (imL-filterL+1.)/recfield
	fc2 = FCLayer(input=conv1.output(), n_in=nfilter1*outL*outL*3, n_out=imL*imL, actfun=sigmoid, tag='_fc2')
	params_cmb = conv1.params + fc2.params 
	#params_cmb = fc0.params + fc2.params
	ypred = fc2.output().reshape((bs, imL, imL))

	model = GeneralModel(input=x, output=ypred,
				target=y, params=params_cmb, 
				regularizers = 0,
				cost_func=mean_sqr_tmp,
				error_func=mean_sqr_tmp)

	sgd = sgd_optimizer(data = msra,  
					model = model,
					batch_size=bs, 
					learning_rate=0.1,
					n_epochs=200)
	sgd.fit()
