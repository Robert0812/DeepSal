# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test1.ui'
#
# Created: Thu Feb  6 20:20:48 2014
#      by: Rui Zhao, The Chinese University of Hong Kong
#          Email: rzhao@ee.cuhk.edu.hk
#
# WARNING! All changes made in this file will be lost!

import os
import sys
from glob import glob 
import numpy as np
import cPickle

from PyQt4.QtGui import *
from PyQt4.QtCore import * 

import string 
import random 

from sals.utils.ImageHelper import *

try: 
    from qimage2ndarray import *
except ImportError:

    def rgb_view(qimage):
        '''
        Convert QImage into a numpy array
        '''
        qimage = qimage.convertToFormat(QImage.Format_RGB32)

        w = qimage.width()
        h = qimage.height()

        ptr = qimage.constBits()
        arr = np.array(ptr).reshape(h, w, 4)
        arr = arr[...,:3]
        arr = arr[:, :, [2, 1, 0]]
        return arr

def array2qimage(rgb):
    """Convert the 3D np array `rgb` into a 32-bit QImage.  `rgb` must
    have three dimensions with the vertical, horizontal and RGB image axes.

    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying np array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!)."""
    if len(rgb.shape) != 3:
        raise ValueError("rgb2QImage can only convert 3D arrays")
    if rgb.shape[2] not in (3, 4):
        raise ValueError("rgb2QImage can expects the last dimension to contain exactly three (R,G,B) or four (R,G,B,A) channels")

    h, w, channels = rgb.shape

    # Qt expects 32bit BGRA data for color images:
    bgra = np.empty((h, w, 4), np.uint8, 'C')
    bgra[...,0] = rgb[...,2]
    bgra[...,1] = rgb[...,1]
    bgra[...,2] = rgb[...,0]
    if rgb.shape[2] == 3:
        bgra[...,3].fill(255)
    else:
        bgra[...,3] = rgb[...,3]

    fmt = QImage.Format_ARGB32
    result = QImage(bgra.data, w, h, fmt)
    result.ndarray = bgra
    return result

class CLabel(QLabel):

    def __init__(self, parent):
        QLabel.__init__(self, parent)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        
        self.emit(SIGNAL('clicked()'))
        event.accept()

class viewer(QWidget):

    def __init__(self, imfiles = None, fpath = None):
        super(viewer, self).__init__()

        if imfiles is None:
            if fpath is None: 
                fpath = '../../../reid_jrnl/salgt/data/gallery'
            imfolder = QFileDialog.getExistingDirectory(None,
                'Select path', fpath, QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
            
            imfiles = glob(str(imfolder) + '/*.bmp')
            imfiles.sort()

        self.imfiles = imfiles

        self.initWin()

    def initWin(self):

        self.setGeometry(QRect(30, 10, 700, 820))
        self.centralwidget = QWidget(self)
        self.imlabel = QLabel(self.centralwidget)
        self.imlabel.setGeometry(QRect(375, 30, 270, 720))

        self.list = QListWidget(self.centralwidget)
        self.list.setGeometry(QRect(10, 30, 300, 720))
        for f in self.imfiles:
            self.list.addItem(os.path.basename(f))

        self.widget1 = QWidget(self.centralwidget)
        self.widget1.setGeometry(QRect(320, 730, 380, 100))
        self.horizontalLayout = QHBoxLayout(self.widget1)
        self.btn_prev = QPushButton(self.widget1)
        self.btn_next = QPushButton(self.widget1)
        self.btn_prev.setText('Prev')
        self.btn_next.setText('Next')
        self.horizontalLayout.addWidget(self.btn_prev)
        self.horizontalLayout.addWidget(self.btn_next)

        QObject.connect(self.btn_next, SIGNAL("clicked()"), self.slot_next)
        QObject.connect(self.btn_prev, SIGNAL("clicked()"), self.slot_prev)

        self.index = 0
        #qimage = QPixmap(self.imfiles[self.index])
        #self.imlabel.setPixmap(qimage.scaled(self.imlabel.size(), Qt.KeepAspectRatio))
        self.show_qpixmap(self.index, self.imlabel)
        self.setWindowTitle('Image Viewer')
        self.show()


    def slot_prev(self):
        ''' previous button '''
        self.index = max(self.index - 1, 0)
        #qimage = QPixmap(self.imfiles[self.index])
        #self.imlabel.setPixmap(qimage.scaled(self.imlabel.size(), Qt.KeepAspectRatio))
        self.show_qpixmap(self.index, self.imlabel)

    def slot_next(self):
        ''' next button '''
        self.index = min(self.index + 1, len(self.imfiles)-1)
        #self.imlabel.setPixmap(qimage.scaled(self.imlabel.size(), Qt.KeepAspectRatio))
        self.show_qpixmap(self.index, self.imlabel)

    def show_qpixmap(self, fidx, qlabel):
        '''
            show a QPixmap to a QLabel 
        '''
        qpixmap = QPixmap(self.imfiles[fidx])
        qimage = qpixmap.toImage()
        imgarr = rgb_view(qimage)
        draw0 = imresize(imgarr, (qlabel.height(), qlabel.width()), interp='bicubic')

        draw1 = draw0 #self.customized_function(draw0, fidx)
         
        qimage = array2qimage(draw1)
        qpixmap_new = QPixmap.fromImage(qimage)
        qlabel.setPixmap(qpixmap_new.scaled(qlabel.size(), Qt.KeepAspectRatio))      

    def customized_function(self, draw0, fidx):
        '''
            customized function for processing the image data 
        '''
        for partid in self.data['scores'][fidx].keys():
            idx = self.data['labels'][fidx] != partid
            diclabel = self.data['scores'][fidx][partid]
            for i in range(2, 3):
                draw0[:, :, i][idx] = np.round(diclabel[0]/(1+diclabel[1])*255.).astype(np.uint8)

        return draw0    

def main():

    app = QApplication(sys.argv)
    mw = viewer()
    mw.show()
    sys.exit(app.exec_())

if __name__ == '__main__':

    main()