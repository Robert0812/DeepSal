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

	def __init__(self):
		''' initialization and parameter settings '''

		super(DataMan_viper, self).__init__()
		self.patL = 48
		self.step = 12

		self.make_data()

	def _load_images_salmaps(self, datapath=None, imgext='bmp'):
		''' load preliminary data (images, segmentations, and salience maps) '''

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

		# imgs_norm = [imnormalize(im) for im in imgs]
		# return imgs, segmsks, salmsks
		self.imgs = imgs 
		self.segmsks = segmsks
		self.salmsks = salmsks


	def _splitting_data(self, ntotal):
		''' generate randomly splitted training / testing data indices ''' 

		randidx = np.random.permutation(ntotal)
		train_num = np.int(0.9*ntotal)
		train_idx = randidx[:train_num]
		test_idx = randidx[train_num:]
	
		return train_idx, test_idx


	def _gen_grids(self):
		''' generate a grid of points for sampling '''
		
		h, w = self.segmsks[0].shape
		_x = np.arange(0, w, self.step)
		_y = np.arange(0, h, self.step)
		self.xx = np.tile(_x, (len(_y), 1)).flatten()
		self.yy = np.tile(_y, (len(_x), 1)).transpose().flatten()


	def _sampling_patch(self, spidx, augx=0):
		''' sampling local patches for predicting salience score '''

		spf = lambda items, indices: map(lambda i: items[i], indices)
		imgs 		=  spf(self.imgs, spidx)
		segmsks 	=  spf(self.segmsks, spidx)
		salmsks 	=  spf(self.salmsks, spidx)

		h, w = self.segmsks[0].shape
		# l = np.int(self.patL/2.)
		# sz = (self.patL, self.patL)
		l = self.patL
		r = np.int(l/2)

		imgids = [] 		# image indices
		ctrids = []			# center grid indices
		patches = [] 		# cropped patches
		salscores = [] 		# corresponding salience scores

		for idx, img, seg, sal in zip(spidx, imgs, segmsks, salmsks):
			
			print idx
			ids1 = []
			ids2 = []
			for i in range(len(self.xx)):
				x = self.xx[i]
				y = self.yy[i]
				if x-r>=0 and x+r<w and y-r>=0 and y+r<h and seg[y, x] > 0 :
					ids1.append(idx)
					ids2.append(i)				

			# crop patches centering on these points
			imgids += ids1
			ctrids += ids2
			patches += [imcrop(img, [self.xx[k]-r, self.yy[k]-r, l, l]) for k in ids2]
			salscores += [np.mean(imcrop(sal, [self.xx[k]-r, self.yy[k]-r, l, l])) for k in ids2]

		# convert image to data format (normalize & roll axis) that is appropriate for training usage
		patches_norm = self._convert_data(patches)
		salscores = np.asarray(salscores)
		imgids = np.asarray(imgids)
		ctrids = np.asarray(ctrids)

		return imgids, ctrids, patches, patches_norm, salscores 


	def _convert_data(self, imgs):
		''' this contains operations on images that make image become input data (cannot be shown) '''
		
		# convert from RGB to CIE-LAB color space
		imgs_lab = [imnormalize(im) for im in imgs]
		# roll axis to put channel axis to the first
		imgs_cvt = [im.transpose((2, 0, 1)) for im in imgs_lab]
		# flatten images
		imgs_arr = np.asarray(imgs_cvt)
		imgs_flatten = imflatten(imgs_arr)
		# normalize to be zero mean and unit std
		imgs_norm = normalize01(imgs_flatten)

		return imgs_norm


	def _shared_dataset(self, data_xy):

		data_x, data_y = data_xy

		shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
		shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=True)
		
		return shared_x, shared_y


	def _share2gpumem(self, data):
		''' share current data into GPU memory '''

		print 'sharing data into GPU memory ...'
		train_set, valid_set, test_set = data
		self.test_x, self.test_y = self._shared_dataset(test_set)
		self.valid_x, self.valid_y = self._shared_dataset(valid_set)
		self.train_x, self.train_y = self._shared_dataset(train_set)


	def make_data(self):
		''' make dataset for training '''

		# read images
		print 'reading ...'
		self._load_images_salmaps()
		
		# splitting training / testing data
		train_idx, test_idx = self._splitting_data(len(self.imgs))

		# preprocessing
		print 'sampling patches from each image ...'
		self._gen_grids()
		self.train_imgids, self.train_ctrids, self.train_ims, train_x, train_y = self._sampling_patch(train_idx)
		self.test_imgids, self.test_ctrids, self.test_ims, test_x, test_y = self._sampling_patch(test_idx)

		# split into train and valid
		nValid = np.int(len(train_x)*0.1)
		train = [train_x[:-nValid], train_y[:-nValid]]
		valid = [train_x[-nValid:], train_y[-nValid:]]
		
		test = [test_x, test_y]
		data = [train, valid, test]
		self._share2gpumem(data)

	

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

	