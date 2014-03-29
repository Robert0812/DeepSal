import theano.tensor as T
from skimage.filter import threshold_otsu

# activation functions

tanh = T.tanh

sigmoid = T.nnet.sigmoid

softmax = T.nnet.softmax

def rectifier(x):
    return x * (x > 0.0)

# cost functions
def mean_cross_entropy(output, target):
	return T.nnet.binary_crossentropy(output, target).mean()

def mean_nneq_map(output, target):
	thresh = threshold_otsu(output)
	return T.neq(1.0*(output>thresh), target).mean()

def mean_sqr(output, target):
    return ((output - target) ** 2).sum(axis=1).mean()

def mean_nll(output, target):
    return -T.mean(T.log(output)[T.arange(target.shape[0]), target])

def mean_nneq(output, target):
    pred = T.argmax(output, axis=1)
    return T.neq(pred, target).mean()

def mean_sqr_tmp(output, target):
	return ((output - target)**2).sum(axis=2).sum(axis=1).mean()/(48*48)