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

    # fig = plt.figure()
    # plotwindow = fig.add_subplot(111)
    # plt.imshow(np.array(img_grey))
    # plt.show()
    img_bin = [0 for i in range(0,784)]

    for i in range(0,28):
        for j in range(0,28):
            if img_grey[i][j] > 136:
                img_bin[i*28+j] = 1

    input_data = img_bin
    output_data = NN.test(input_data)
    out_act = output_data.index(max(output_data))

    print "The image is recognized as :" + str(out_act)
    #test
    # dataset = minist.minist()
    # hit = 0
    # dataset.test_init()
    # test_data_num = 10000
    # test_error = 0
    # for j in range(0,test_data_num):
    #
    #     im = dataset.get_test_image()
    #     input_data = [0 for i in range(0,784)]
    #
    #     for k in range(0,784):
    #         if im[k]>100:
    #             input_data[k] = 1
    #
    #     om = dataset.get_test_label()
    #     print "This test sample is :" + str(om)
    #
    #     output_data = NN.test(input_data)
    #     out_act = output_data.index(max(output_data))
    #     print "The actual output is :" + str(out_act)
    #     if om == out_act:
    #         hit += 1
    #
    #     test_error += NN.test_error**2
    #
    # print "The test error is :" + str(test_error/(2*test_data_num))
    # print "The hit rate is :" + str(hit/10000.0)
