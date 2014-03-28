import gzip
import cPickle
import os 
import numpy as np 
from scipy.io import loadmat, savemat
import theano 
import theano.tensor as T
from sals.utils.ImageHelper import imresize
import glob 
import ntpath
import pylab as pl 

class DataMan(object):

	def __init__(self, filepath=None):

		if filepath is not None:
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

		self._log('Saving file to {}'.format(savefile))

		if savefile[-3:] == 'pkl':
			f = open(savefile, 'wb')
			print savefile, len(data)
			cPickle.dump(data, f, -1) 
			f.close()

		elif savefile[-3:] == 'csv':
			with open(savefile, 'wb') as f:
				w = csv.writer(f)
				w.writerows(data)

		elif savefile[-3:] == 'mat':
			savemat(savefile, data)


class DataMan_mnist(DataMan):

	def __init__(self, filepath=None):
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

		print 'sharing data into GPU memory ...'
		train_set, valid_set, test_set = data
		self.test_x, self.test_y = self.shared_dataset(test_set)
		self.valid_x, self.valid_y = self.shared_dataset(valid_set)
		self.train_x, self.train_y = self.shared_dataset(train_set)


class DataMan_msra(DataMan):

	def __init__(self, filepath=None):
		super(DataMan_msra, self).__init__(filepath)


	def convert2pkl(self, pklfile, sz=(48, 48)):

		if not os.path.isfile(pklfile):
			dataset_dir = '/home/rzhao/Projects/deep-saliency/data/'
			thus10000 = dataset_dir + 'THUS10000_Imgs_GT/Imgs'
			msra5000 = dataset_dir + 'MSRA5000/Imgs'
			msra5000_test = dataset_dir + 'MSRA5000/MSRA-B-test'
			img_ext = '.jpg'
			msk_ext = '.png'

			trn_img = []
			trn_msk = []
			for single_image in sorted(glob.glob(thus10000+'/*'+img_ext)):
				rsb = glob.glob(msra5000_test+'/*_'+ntpath.basename(single_image)[:-4]+'_smap'+msk_ext)
				if len(rsb) == 0:
					trn_img.append(single_image)
					trn_msk.append(single_image[:-4]+msk_ext)

			tst_img = []
			tst_msk = []
			for single_image in sorted(glob.glob(msra5000_test+'/*'+msk_ext)):
				tst_img.append(msra5000+'/'+ntpath.basename(single_image)[:-len('_smap.png')]+img_ext)	
				tst_msk.append(msra5000+'/'+ntpath.basename(single_image)[:-len('_smap.png')]+msk_ext)

			improc_func = lambda im: imresize(im, sz, interp='bicubic') 

			train_x = [improc_func(pl.imread(fname)) for fname in trn_img]
			train_y = [improc_func(pl.imread(fname)) for fname in trn_msk]

			np.random.seed(123)
			np.random.shuffle(train_x)
			np.random.seed(123)
			np.random.shuffle(train_y)

			test_x = [improc_func(pl.imread(fname)) for fname in tst_img]
			test_y = [improc_func(pl.imread(fname)) for fname in tst_msk]

			train = [[train_x[id] for id in range(7000)], [train_y[id] for id in range(7000)]]
			valid = [[train_x[id] for id in range(7001, len(train_x))], [train_y[id] for id in range(7001, len(train_y))]]
			test = [test_x, test_y]

			data = [train, valid, test] 
			self.save(data, pklfile)

		else:
			print 'History pickle file exists!'


	def shared_dataset(self, data_xy):

		data_x, data_y = data_xy

		shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
		shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=True)
		
		return shared_x, T.cast(shared_y, 'int32')


	def share2gpumem(self, data):
		''' share current data into GPU memory '''
		print 'sharing data into GPU memory ...'
		train_set, valid_set, test_set = data
		self.test_x, self.test_y = self.shared_dataset(test_set)
		self.valid_x, self.valid_y = self.shared_dataset(valid_set)
		self.train_x, self.train_y = self.shared_dataset(train_set)