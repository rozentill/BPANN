#! /usr/env/bin
#-*-coding:utf8-*-

import numpy as np
import struct
import matplotlib.pyplot as plt

class minist(object):

    def __init__(self):

        self.train_image_file = None
        self.train_label_file = None

        self.train_image = None
        self.train_label = None
        self.train_image_index = 0
        self.train_label_index = 0

        self.test_image = None
        self.test_label = None
        self.test_image_index = 0
        self.test_label_index = 0

    def train_init(self):

        self.test_image_index = 0
        self.test_label_index = 0

        self.train_image_file = open('./MINIST/train-images-idx3-ubyte')
        self.train_image = self.train_image_file.read()
        self.train_label_file = open('./MINIST/train-labels-idx1-ubyte')
        self.train_label = self.train_label_file.read()

        # omit the header lines of the files
        magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , self.train_image , self.train_image_index)
        magic, numItem = struct.unpack_from('>II' , self.train_label , self.train_label_index)

        self.train_image_index += struct.calcsize('>IIII')
        self.train_label_index += struct.calcsize('>II')

    def get_train_image(self):
        # get a data sample
        im = struct.unpack_from('>784B' ,self.train_image, self.train_image_index)
        self.train_image_index += struct.calcsize('>784B')

        return im#a tuple

    def get_train_label(self):

        im = struct.unpack_from('>1B' , self.train_label, self.train_label_index)
        self.train_label_index += struct.calcsize('>1B')

        return im[0]

    def train_finished(self):

        self.train_image_file.close()
        self.train_label_file.close()


    def test_init(self):

        self.test_image = open('./MINIST/t10k-images-idx3-ubyte').read()
        self.test_label = open('./MINIST/t10k-labels-idx1-ubyte').read()

        # omit the header lines of the files
        magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , self.test_image , self.test_image_index)
        magic, numItem = struct.unpack_from('>II' , self.test_label , self.test_label_index)

        self.test_image_index += struct.calcsize('>IIII')
        self.test_label_index += struct.calcsize('>II')

    def get_test_image(self):

        # get a data sample
        im = struct.unpack_from('>784B' ,self.test_image, self.test_image_index)
        self.test_image_index += struct.calcsize('>784B')

        return im#a tuple

    def get_test_label(self):

        im = struct.unpack_from('>1B' , self.test_label, self.test_label_index)
        self.test_label_index += struct.calcsize('>1B')

        return im[0]
