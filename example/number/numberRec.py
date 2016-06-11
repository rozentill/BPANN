#! /usr/env/bin
#-*-coding:utf8-*-
import net
import numpy as np
import minist


if __name__ == '__main__':

    train_data_num = 60000
    test_data_num = 10000
    iteration = 1
    rate = 0.05
    train_error = 0
    test_error = 0

    # Define a neural network
    NN = net.BPANN(784,17,10,1,rate)

    # train
    dataset = minist.minist()

    for i in range(1,iteration+1):
        dataset.train_init()
        train_error = 0
        for j in range(0,train_data_num):

            im = dataset.get_train_image()
            input_data = [0 for i in range(0,784)]

            for k in range(0,784):
                if im[k]>100:
                    input_data[k] = 1

            om = dataset.get_train_label()
            print "Get a sample :" + str(om)
            output_data = [-1 for i in range(0,10)]
            output_data[om] = 1

            NN.train(input_data,output_data)

            train_error += NN.train_error**2

        print "This iteration's error is :" + str(train_error/(2*train_data_num))
        dataset.train_finished()

    #test
    hit = 0
    dataset.test_init()

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
