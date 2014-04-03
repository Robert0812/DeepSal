import theano.tensor as T
from skimage.filter import threshold_otsu
from sklearn.decomposition import PCA

''' preprocessing functions '''
def normalize01(X):
	''' zeros mean and unit standard deviation '''
	X -= X.mean(axis=0)
	X /= X.std(axis=0)
	return X

def PCA_whitening(X_train, X_test, n_dim=None):
	''' PCA whitening '''
	if n_dim is None:
		n_dim = X_train.shape[1]
	print 'X_train:{}, X_test:{}'.format(X_train.shape, X_test.shape)
	pca = PCA(n_components=n_dim, whiten=True)
	pca.fit(X_train)
	return [pca.transform(X_train), pca.transform(X_test)]

def flatten(X):
	''' flatten data from axis 1 to end '''
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

def mean_nneq_map(output, target):
	thresh = threshold_otsu(output)
	return T.neq(1.0*(output>thresh), target).sum(axis=1).mean()

def mean_sqr(output, target):
    return ((output - target) ** 2).sum(axis=1).mean()

def mean_nll(output, target):
    return -T.mean(T.log(output)[T.arange(target.shape[0]), target])

def mean_nneq(output, target):
    pred = T.argmax(output, axis=1)
    return T.neq(pred, target).mean()


''' evaluation functions for saliency map '''

def get_confusion(gtmask, dtmask):
    cfm = np.zeros((2, 2))
    neg_gtmask = 1 - gtmask
    neg_dtmask = 1 - dtmask
    cfm[0, 0] = (neg_gtmask * neg_dtmask).sum() #tn
    cfm[0, 1] = (neg_gtmask * dtmask).sum()     #fp
    cfm[1, 0] = (gtmask * neg_dtmask).sum()     #fn
    cfm[1, 1] = (gtmask * dtmask).sum()         #tp
    return cfm

def get_precision(confusion):
    """
    >>> precision([[5,1],[1,1]])
    0.5
    """
    judged_pos = confusion[0][1] + confusion[1][1]
    if judged_pos == 0:
        return 0.0
    return (confusion[1][1]/
            float(judged_pos))

def get_recall(confusion):
    """
    >>> recall([[5,1],[1,3]])
    0.75
    """
    pos = confusion[1][0] + confusion[1][1]
    if pos == 0:
        return 0.0
    return (confusion[1][1]/
            float(pos))

def get_fpr(confusion):
    return confusion[0][1]/(confusion[0][1] + confusion[0][0])

def get_tpr(confusion):
    return confusion[1][1]/(confusion[1][1] + confusion[1][0])

def get_roc(true_mask, esti_mask):
    cfm = get_confusion(true_mask, esti_mask)
    prec = get_precision(cfm)
    reca = get_recall(cfm)
    return reca, prec

def get_fbeta(true_mask, esti_mask, beta2 = 0.3):
    confusion = get_confusion(true_mask, esti_mask)
    p = get_precision(confusion)
    r = get_recall(confusion)
    if p + r == 0.0:
        return 0.0    
    fscore = (1 + beta2) * (p * r) / (
        beta2 * p + r)
    return fscore