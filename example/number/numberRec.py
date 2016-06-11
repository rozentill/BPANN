#! /usr/env/bin
#-*-coding:utf8-*-

import net
import numpy as np
import minist

if __name__ == "__main__":

    # Define a neural network
    NN = net.BPANN(784,17,10)
    NN.loadModel("model.txt")

    #test
    dataset = minist.minist()
    hit = 0
    dataset.test_init()
    test_data_num = 10000
    test_error = 0
    for j in range(0,test_data_num):

        im = dataset.get_test_image()
        input_data = [0 for i in range(0,784)]

        for k in range(0,784):
            if im[k]>100:
                input_data[k] = 1

        om = dataset.get_test_label()
        print "This test sample is :" + str(om)

        output_data = NN.test(input_data)
        out_act = output_data.index(max(output_data))
        print "The actual output is :" + str(out_act)
        if om == out_act:
            hit += 1

        test_error += NN.test_error**2

    print "The test error is :" + str(test_error/(2*test_data_num))
    print "The hit rate is :" + str(hit/10000.0)
