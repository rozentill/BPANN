#! /usr/env/bin
#-*-coding:utf8-*-

import random
import numpy as np

class BPANN(object):

    def __init__(self, ni, nh, no, funcNo):

        #initialize the number of nodes
        #ni means the number of features of input
        self.numInput = ni
        self.numHidden = nh
        self.numOutput = no

        #initialize the weight matrices
        self.weightInputHidden = np.array([[random.uniform(0,0.2) for x in range(1,nh+1)] for y in range(1,ni+1)])
        self.weightHiddenOutput = np.array([[random.uniform(0,4)-2 for x in range(1,no+1)] for y in range(1,nh+1)])

        #initialize the biases
        self.biasHidden = np.array([random.uniform(0,0.2) for x in range(1,nh+1)])
        self.biasOutput = np.array([random.uniform(0,4)-2 for x in range(1,no+1)])

        #initialize the outputs of each layer, as a0, a1, a2
        self.vecInput = np.array（[0 for x in range(1,ni+1)]）
        self.vecHidden = np.array([0 for x in range(1,nh+1)])
        self.vecOutput = np.array([0 for x in range(1,no+1)])

        #initialize sensitivity
        senOutput = np.array([0 for x in range(1,no+1)])
        senHidden = np.array([0 for x in range(1,nh+1)])

        #initialize error
        train_error = 0
        test_error = 0

        #choose activation function


    def feedForward(self):
        self.vecHidden = map(self.activation,np.ndarray.tolist(np.dot(self.vecInput,self.weightInputHidden)+self.biasHidden))

    def backPropagate(self,input_data):

    # train one sample ,input is a list
    def train(self,input_data):
        self.vecInput = np.array(input_data)
        feedForward()
