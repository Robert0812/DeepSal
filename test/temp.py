import pylab as pl
from skimage import color
from skimage import exposure
from sklearn.cluster import KMeans
from scipy.stats import itemfreq
import numpy as np

K = 10

im = pl.imread('/home/rzhao/Downloads/1.jpg')
img_eq = exposure.equalize_hist(im)
im_lab = color.rgb2lab(img_eq)
data = im_lab.transpose((2, 0, 1))
data0 = data.reshape((3, np.prod(im.shape[0:2])))
kmeans = KMeans(init='k-means++', n_clusters=K, n_init=1)
kmeans.fit(data0.T)
mask = kmeans.labels_.copy()

pl.figure(1)
for i in range(K):
	mask0 = mask.copy()
	mask0[mask==i] = 255
	pl.subplot(1, K, i)
	pl.imshow(mask0.reshape((im.shape[0], im.shape[1])))

pl.show()