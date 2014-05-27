import gzip
import cPickle
import os 
import sys
import numpy as np 
from scipy.io import loadmat, savemat
import theano 
import theano.tensor as T

from glob import glob 
import ntpath
import pylab as pl 

from sals.utils.ImageHelper import imnormalize, imcrop, imflatten, imread, imresize
from sals.utils.FunctionHelper import normalize01
from sals.utils.utils import *

class DataMan(object):

	def __init__(self, filepath=None):

		if filepath is not None:
			if os.path.isfile(filepath):
				self._file = filepath 
			else:
				raise IOError('File not exist')


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


	def _log(self, msg):
		''' Print verbose information '''
		print msg


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


class DataMan_viper(DataMan):

	def __init__(self, filepath=None):
		super(DataMan_viper, self).__init__(filepath)


	def shared_dataset(self, data_xy):

		data_x, data_y = data_xy

		shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
		shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=True)
		
		return shared_x, shared_y


	def share2gpumem(self, data):
		''' share current data into GPU memory '''
		print 'sharing data into GPU memory ...'
		train_set, valid_set, test_set = data
		self.test_x, self.test_y = self.shared_dataset(test_set)
		self.valid_x, self.valid_y = self.shared_dataset(valid_set)
		self.train_x, self.train_y = self.shared_dataset(train_set)


	def load_images_salmaps(self, datapath=None, imgext='bmp'):

		if datapath is None:
			if sys.platform == 'darwin':
				homedir = '/Users/rzhao/'
			else:
				homedir = '/home/rzhao/'

			datapath = homedir + 'Dropbox/ongoing/reid_jrnl/salgt/data_viper/'

		filepath = datapath + 'query/'
		imgfiles = sorted(glob(filepath + '*.' + imgext))
		imgs = [imread(im) for im in imgfiles]

		salfilepath = datapath + 'labels.pkl'
		data = loadfile(salfilepath)
		segmsks, salmsks = data
		
		imgs = [imresize(im, size=(segmsks[0].shape), interp='bicubic') for im in imgs]

		imgs_norm = [imnormalize(im) for im in imgs]
		return imgs_norm, segmsks, salmsks


	def convert_data(self, imgs):
		imgs_norm = [im.transpose((2, 0, 1)) for im in imgs]
		return imgs_norm


	def sampling_patch(self, imgs, segmsks, salmsks, sz=(48,48), ns= 1000, augx=0):

		h, w = segmsks[0].shape
		l = np.int(sz[0]/2.)
		xx = np.tile(np.asarray(range(w)), (h, 1)).flatten()
		yy = np.tile(np.asarray(range(h)), (w, 1)).transpose().flatten()

		imgs_crop = []
		sals_crop = []
		
		for img, seg, sal in zip(imgs, segmsks, salmsks):
			# generate random sampling entries
			samplemsk = np.zeros(h*w)
			randidx = np.random.permutation(h*w)
			samplemsk[randidx[:ns]] = 1
			samplemsk = samplemsk.reshape((h, w))

			toplefts = []
			for x, y in zip(xx, yy):
				if x-l>=0 and x+l<w and y-l>=0 and y+l<h and seg[y, x] > 0 and samplemsk[y, x] > 0:
					toplefts.append((np.int(x-l), np.int(y-l)))

			# crop patches centering on these points
			imgs_crop += [imcrop(img, np.hstack((pt, sz))) for pt in toplefts]
			sals_crop += [np.mean(imcrop(sal, np.hstack((pt, sz)) )) for pt in toplefts]

		# convert data to roll axis for training usage
		imgs_norm = self.convert_data(imgs_crop)
		# sals_norm = np.asarray(sals_crop)

		return imgs_norm, sals_crop


	def splitting_data(self, ntotal):
	
		randidx = np.random.permutation(ntotal)
		train_num = np.int(0.9*ntotal)
		train_idx = randidx[:train_num]
		test_idx = randidx[train_num:]
	
		return train_idx, test_idx


	def convert2pkl(self, pklfile):

		if not os.path.isfile(pklfile):

			# read images
			print 'reading ...'
			imgs, segmsks, salmsks = self.load_images_salmaps()
			
			# splitting training / testing data
			train_idx, test_idx = self.splitting_data(len(imgs))

			# preprocessing
			print 'sampling patches from each image ...'
			spf = lambda items, indices: map(lambda i: items[i], indices)
			train_x, train_y = self.sampling_patch(spf(imgs, train_idx), spf(segmsks, train_idx), spf(salmsks, train_idx))
			test_x, test_y = self.sampling_patch(spf(imgs, test_idx), spf(segmsks, test_idx), spf(salmsks, test_idx))

			# shuffle training data
			print 'shuffle data ...'
			np.random.seed(123)
			np.random.shuffle(train_x)
			np.random.seed(123)
			np.random.shuffle(train_y)

			# flattern and dtype conversion
			print 'flatten data ...'
			train_x = np.asarray(train_x, dtype=np.float32)
			train_y = np.asarray(train_y, dtype=np.float32)
			test_x = np.asarray(test_x, dtype=np.float32)
			test_y = np.asarray(test_y, dtype=np.float32)
			train_x = imflatten(train_x)
			# train_y = flatten(train_y)
			test_x = imflatten(test_x)
			# test_y = flatten(test_y)

			# normalize data to have zero mean and unit std
			train_x = normalize01(train_x)
			test_x = normalize01(test_x)

			# split into train and valid
			nValid = np.int(len(train_x)*0.1)
			train = [train_x[:-nValid], train_y[:-nValid]]
			valid = [train_x[-nValid:], train_y[-nValid:]]
			
			test = [test_x, test_y]
			data = [train, valid, test]
			savefile(data, pklfile)

		else:
			print 'History pickle file exists!'


class DataMan_msra(DataMan):

	def __init__(self, filepath=None):
		super(DataMan_msra, self).__init__(filepath)


	def shared_dataset(self, data_xy):

		data_x, data_y = data_xy

		shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
		shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=True)
		
		return shared_x, shared_y

		#return shared_x, T.cast(shared_y, 'int32')


	def share2gpumem(self, data):
		''' share current data into GPU memory '''
		print 'sharing data into GPU memory ...'
		train_set, valid_set, test_set = data
		self.test_x, self.test_y = self.shared_dataset(test_set)
		self.valid_x, self.valid_y = self.shared_dataset(valid_set)
		self.train_x, self.train_y = self.shared_dataset(train_set)


	def convert_data(self, imgs, msks):
		imgs_norm = [im.transpose((2, 0, 1)) for im in imgs]
		msks_norm = [(im>127)*1.0 for im in msks]
		return imgs_norm, msks_norm


	def preprocessing(self, imgs, msks, sz=(48, 48), augx=0):

		print len(imgs)
		if augx > 0:
			print 'augmenting train data ...'			
			# augx = 2xnSample+1
			n_sample = np.int(augx/2.)-1
			imH, imW = imgs[0].shape[0:2]
			borderH = np.int(imH*0.2)
			borderW = np.int(imW*0.2) 
			w = imW - borderW 
			h = imH - borderH
			x1s = np.random.randint(0, borderW, n_sample)
			y1s = np.random.randint(0, borderH, n_sample)
			imgs_crop = imgs
			msks_crop = msks
			for img, msk in zip(imgs, msks):
				imgs_crop += [imcrop(img, [x1, y1, w, h]) for x1, y1 in zip(x1s, y1s)]
				msks_crop += [imcrop(msk, [x1, y1, w, h]) for x1, y1 in zip(x1s, y1s)]
			print len(imgs_crop)
			imgs_flip = [pl.fliplr(im) for im in imgs_crop]
			msks_flip = [pl.fliplr(im) for im in msks_crop]
			imgs = imgs_crop + imgs_flip
			msks = msks_crop + msks_flip 
			print len(imgs)
			
		imgs_rs = [imresize(im, sz, interp='bicubic') for im in imgs]
		imgs_norm = [imnormalize(im) for im in imgs_rs]
		msks_norm = [imresize(im, sz, interp='bicubic') for im in msks]
		imgs_final, msks_final = self.convert_data(imgs_norm, msks_norm)
		print len(imgs_final)
		return imgs_final, msks_final

	def convert2pkl(self, pklfile):

		if not os.path.isfile(pklfile):
			dataset_dir = '/home/rzhao/Projects/deep-saliency/data/'
			thus10000 = dataset_dir + 'THUS10000_Imgs_GT/Imgs'
			msra5000 = dataset_dir + 'MSRA5000/Imgs'
			msra5000_test = dataset_dir + 'MSRA5000/MSRA-B-test'
			img_ext = '.jpg'
			msk_ext = '.png'
			augX = 10

			trn_img = []
			trn_msk = []
			for single_image in sorted(glob(thus10000+'/*'+img_ext)):
				rsb = glob(msra5000_test+'/*_'+ntpath.basename(single_image)[:-4]+'_smap'+msk_ext)
				if len(rsb) == 0:
					trn_img.append(single_image)
					trn_msk.append(single_image[:-4]+msk_ext)

			tst_img = []
			tst_msk = []
			for single_image in sorted(glob(msra5000_test+'/*'+msk_ext)):
				tst_img.append(msra5000+'/'+ntpath.basename(single_image)[:-len('_smap.png')]+img_ext)	
				tst_msk.append(msra5000+'/'+ntpath.basename(single_image)[:-len('_smap.png')]+msk_ext)

			# read images
			print 'reading ...'
			train_img = [imread(fname) for fname in trn_img]
			train_msk = [imread(fname) for fname in trn_msk]
			test_img = [imread(fname) for fname in tst_img]
			test_msk = [imread(fname) for fname in tst_msk]

			# preprocessing
			print 'preprocessing ...'			
			train_x, train_y = self.preprocessing(train_img, train_msk, augx=augX)
			test_x, test_y = self.preprocessing(test_img, test_msk, augx=0)

			# shuffle training data
			print 'shuffle data ...'
			np.random.seed(123)
			np.random.shuffle(train_x)
			np.random.seed(123)
			np.random.shuffle(train_y)

			# flattern and dtype conversion
			print 'flatten data ...'
			train_x = np.asarray(train_x, dtype=np.float32)
			train_y = np.asarray(train_y, dtype=np.float32)
			test_x = np.asarray(test_x, dtype=np.float32)
			test_y = np.asarray(test_y, dtype=np.float32)
			train_x = imflatten(train_x)
			train_y = imflatten(train_y)
			test_x = imflatten(test_x)
			test_y = imflatten(test_y)

			# normalize data to have zero mean and unit std
			train_x = normalize01(train_x)
			test_x = normalize01(test_x)

			# split into train and valid
			nValid = np.int(len(train_img)*0.1)*augX
			train = [train_x[:-nValid], train_y[:-nValid]]
			valid = [train_x[-nValid:], train_y[-nValid:]]
			# train = [train_x[0:7000], train_y[0:7000]]
			# valid = [train_x[7000:], train_y[7000:]]
			test = [test_x, test_y]
			data = [train, valid, test]
			self.save(data, pklfile)

		else:
			print 'History pickle file exists!'

	