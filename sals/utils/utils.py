from cStringIO import StringIO
import os
import sys
from glob import glob 
import webbrowser
import gzip
import cPickle
from scipy.io import loadmat

def loadfile(file=None):
	''' Load data from file with different format '''

	if file is None:
		raise NameError('File not specified!')

	print 'Loading file at {}'.format(file)

	if file[-3:] == '.gz':

		f = gzip.open(file, 'rb')
		data = cPickle.load(f)

	elif file[-3:] == 'pkl':
		with open(file, 'rb') as f:
			data = cPickle.load(f)

	elif file[-3:] == 'csv':
		with open(file, 'rb') as f:
			reader = csv.reader(f)
			data = [row for row in reader]

	elif file[-3:] == 'mat':
		data = loadmat(file)

	else:
		raise NameError('File format not recognized')

	return data

def savefile(data, file = None):	
		''' Save data to file '''

		if file is None:
			raise NameError('File not specified!')

		print 'Saving file to {}'.format(file)

		if file[-3:] == 'pkl':
			f = open(file, 'wb')
			cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL) 
			f.close()

		elif file[-3:] == 'csv':
			with open(file, 'wb') as f:
				w = csv.writer(f)
				w.writerows(data)

		elif file[-3:] == 'mat':
			savemat(file, data)
			

def visualize_imfolder(folder_path=None):

	if folder_path is None:
		folder_path = '../data_test/result/'
	else:
		folder_path = folder_path + ('/' if folder_path[-1] is not '/' else '')

	imgfiles = sorted(glob(folder_path + '*.jpg'))
	imgfiles = [os.path.basename(f) for f in imgfiles]

	file_str = StringIO()
	file_str.write("<body><table>\n")

	for imgname in imgfiles:
		file_str.write("<tr>") # new row
		file_str.write("<td>")
		file_str.write(imgname)
		file_str.write("</t d>")
		file_str.write("<td>")
		file_str.write("<img src=\"" + imgname + "\" />")
		file_str.write("</td>")
		file_str.write("</tr>\n")

	file_str.write("</table></body>")
	
	htmlfile = open(folder_path + 'index.html', 'w')
	htmlfile.write(file_str.getvalue())
	htmlfile.close()
	webbrowser.open(folder_path + 'index.html')