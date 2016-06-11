#! /usr/env/bin
#-*-coding:utf8-*-

import net
import numpy as np
import minist
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":

    # Define a neural network
    NN = net.BPANN(784,17,10)
    NN.loadModel("model.txt")

    filename = raw_input("Please enter the file path you want to recognize :")

    img = plt.imread('./numbers/'+filename+'.jpg')
    img_grey = [[0 for i in range(0,28)] for j in range(0,28)]

    for i in range(0,28):
        for j in range(0,28):
            img_grey[i][j] = 0.3*img[i][j][0] + 0.59*img[i][j][1] + 0.11*img[i][j][2]

    img_bin = [0 for i in range(0,784)]

    for i in range(0,28):
        for j in range(0,28):
            if img_grey[i][j] > 136:
                img_bin[i*28+j] = 1

    input_data = img_bin
    output_data = NN.test(input_data)
    out_act = output_data.index(max(output_data))

    print "The image is recognized as :" + str(out_act)
