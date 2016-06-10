#! /usr/env/bin
#-*-coding:utf8-*-

import net
import numpy as np

if __name__ == '__main__':
    nn = net.BPANN(1,2,1)
    nn.weightInputHidden = np.array([[-0.27,-0.41]])
    nn.biasHidden = np.array([[-0.48,-0.13]])
    nn.weightHiddenOutput = np.array([[0.09],[-0.17]])
    nn.biasOutput = np.array([[0.48]])
    input_data = [1]
    output_data = [1.707106781]
    nn.train(input_data,output_data)
    print nn.weightInputHidden
    print nn.biasHidden
    print nn.weightHiddenOutput
    print nn.biasOutput
