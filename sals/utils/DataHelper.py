import gzip
import cPickle
import os 
import numpy as np 
from scipy.io import loadmat, savemat
import theano 
import theano.tensor as T

class DataMan(object):

	def __init__(self, filepath):

		if os.path.isfile(filepath):
			self._file = filepath 
		else:
			raise IOError('File not exist')


	def _log(self, msg):
		''' Print verbose information '''
		print msg


	def load(self):
		''' Load data from file with different format '''

		self._log('Loading file at {}'.format(self._file))

		if self._file[-3:] == '.gz':

			f = gzip.open(self._file, 'rb')
			data = cPickle.load(f)

		elif self._file[-3:] == 'pkl':
			with open(self._file, 'rb') as f:
				data = cPickle.load(f)

		elif self._file[-3:] == 'csv':
			with open(self._file, 'rb') as f:
				reader = csv.reader(f)
				data = [row for row in reader]

		elif self._file[-3:] == 'mat':
			data = loadmat(self._file)

		else:
			raise NameError('File format not recognized')

		return data


	def save(self, data, savefile):	
		''' Save data to file '''

		self._log('Loading file at {}'.format(savefile))

		if savefile[:-3] == 'pkl':
			with open(savefile, 'wb') as f:
				cPickle.dump(data, savefile, cPickle.HIGHEST_PROTOCOL)

		elif savefile[:-3] == 'csv':
			with open(self._file, 'wb') as f:
				w = csv.writer(f)
				w.writerows(data)

		elif savefile[:-3] == 'mat':
			savemat(savefile, data)

class DataMan_mnist(DataMan):

	def __init__(self, filepath):
		super(DataMan_mnist, self).__init__(filepath)
		
	def shared_dataset(self, data_xy):

		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
		shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=True)
		
		# we will have to cast it to int. 
		return shared_x, T.cast(shared_y, 'int32')
		#def group(self, data, batchsize):
		#	return [data[i:i+batchsize] for i in range(0, len(data), batchsize)]

	def share2gpumem(self, data):
		''' share current data into GPU memory '''

		train_set, valid_set, test_set = data
		self.test_x, self.test_y = self.shared_dataset(test_set)
		self.valid_x, self.valid_y = self.shared_dataset(valid_set)
		self.train_x, self.train_y = self.shared_dataset(train_set)
