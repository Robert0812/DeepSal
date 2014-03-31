import theano.tensor as T
from skimage.filter import threshold_otsu
from sklearn.decomposition import PCA

''' preprocessing functions '''
def normalize_data(X_train, X_test, n_dim=None):
	if n_dim is None:
		n_dim = X_train.shape[1]
	print 'X_train:{}, X_test:{}'.format(X_train.shape, X_test.shape)
	pca = PCA(n_components=n_dim, whiten=True)
	pca.fit(X_train)
	return [pca.transform(X_train), pca.transform(X_test)]

def flatten(X):
	n_X = X.shape[0]
	return X.reshape((n_X, -1))


''' activation functions '''
tanh = T.tanh

sigmoid = T.nnet.sigmoid

softmax = T.nnet.softmax

def rectifier(x):
    return x * (x > 0.0)


''' cost functions '''
def mean_cross_entropy(output, target):
	return T.nnet.binary_crossentropy(output, target).sum(axis=1).mean()

def mean_cross_entropy_map(output, target):
	return T.nnet.binary_crossentropy(output, target).sum(axis=1).mean()/(48*48)

def mean_nneq_map(output, target):
	thresh = threshold_otsu(output)
	return T.neq(1.0*(output>thresh), target).sum(axis=1).mean()/(48*48)

def mean_sqr(output, target):
    return ((output - target) ** 2).sum(axis=1).mean()

def mean_nll(output, target):
    return -T.mean(T.log(output)[T.arange(target.shape[0]), target])

def mean_nneq(output, target):
    pred = T.argmax(output, axis=1)
    return T.neq(pred, target).mean()

def mean_sqr_map(output, target):
	return ((output - target)**2).sum(axis=1).mean()/(48*48)