#! /usr/env/bin
#-*-coding:utf8-*-

import random
import numpy as np
import function as func
import string

class BPANN(object):

    def __init__(self, ni, nh, no, funcNo=1, rate=0.1):

        #initialize the number of nodes
        #ni means the number of features of input
        self.numInput = ni
        self.numHidden = nh
        self.numOutput = no

        #initialize the weight matrices
        self.weightInputHidden = np.array([[random.uniform(0,0.2) for x in range(1,nh+1)] for y in range(1,ni+1)])
        self.weightHiddenOutput = np.array([[random.uniform(0,4)-2 for x in range(1,no+1)] for y in range(1,nh+1)])

        #initialize the biases
        self.biasHidden = np.array([[random.uniform(0,0.2) for x in range(1,nh+1)]])
        self.biasOutput = np.array([[random.uniform(0,4)-2 for x in range(1,no+1)]])

        #initialize the outputs of each layer, as a0, a1, a2
        self.vecInput = np.array([[0 for x in range(1,ni+1)]])
        self.vecHidden = np.array([[0 for x in range(1,nh+1)]])
        self.vecOutput = np.array([[0 for x in range(1,no+1)]])
        self.expectOutput = np.array([[0 for x in range(1,no+1)]])

        #initialize sensitivity
        self.senOutput = np.array([[0 for x in range(1,no+1)]])
        self.senHidden = np.array([[0 for x in range(1,nh+1)]])

        #initialize error
        self.train_error = 0
        self.test_error = 0

        #choose activation function
        if funcNo is 1:#sigmoid
            self.activation = func.sigmoid
            self.diff_activation = func.diff_sigmoid
        elif funcNo is 2:
            self.activation = math.tanh
            self.diff_activation = func.diff_tanh

        #learning rate
        self.rate = rate

    def feedForward(self):
        self.vecHidden = np.array([
            map(
                self.activation,
                np.ndarray.tolist(
                    np.dot(self.vecInput,self.weightInputHidden)[0]+self.biasHidden[0]
                )
            )
        ])
        self.vecOutput = np.dot(self.vecHidden,self.weightHiddenOutput)+self.biasOutput

    def backPropagate(self):
        # compute the error
        self.train_error = 0
        for i in range(0,self.numOutput):
            self.train_error += (self.expectOutput[0][i] - self.vecOutput[0][i])
        self.train_error = self.train_error/self.numOutput

        # back propagate from s2
        self.senOutput = -2 * (self.expectOutput - self.vecOutput)

        self.weightHiddenOutput.shape = (self.numHidden,self.numOutput)
        # now update s1
        self.senHidden = np.dot(
            self.senOutput,np.dot(
                np.transpose(
                    self.weightHiddenOutput
                ),
                (1-self.vecHidden) * self.vecHidden * np.eye(self.numHidden)# (1-a)a * I = l*l
            )
        )


    def updateWeight(self):

        #update w2
        self.weightHiddenOutput = self.weightHiddenOutput - self.rate * np.dot(
            np.transpose(
                self.vecHidden,
            ),
            self.senOutput
        )
        #update w1
        self.weightInputHidden = self.weightInputHidden - self.rate * np.dot(
            np.transpose(
                self.vecInput
            ),
            self.senHidden
        )

        #update b2
        self.biasOutput = self.biasOutput - self.rate * self.senOutput

        #update b1
        self.biasHidden = self.biasHidden - self.rate * self.senHidden

    # train one sample ,input is a list
    def train(self,input_data,output_data):
        self.vecInput = np.array([input_data])
        self.expectOutput = np.array([output_data])
        self.feedForward()
        self.backPropagate()
        self.updateWeight()
        return self.train_error


    def test(self,input_data):
        self.vecInput = np.array([input_data])
        self.feedForward()
        return np.ndarray.tolist(self.vecOutput[0])

    def saveModel(self,filename):
        f = open(filename,'w')
        #write weightInputHidden
        for i in range(0,self.numInput):
            for j in range(0,self.numHidden):
                f.write(str(self.weightInputHidden[i][j])+'\n')

        #write weightHiddenOutput
        for i in range(0,self.numHidden):
            for j in range(0,self.numOutput):
                f.write(str(self.weightHiddenOutput[i][j])+'\n')

        #write biasHidden
        for i in range(0,self.numHidden):
            f.write(str(self.biasHidden[0][i])+'\n')

        #write biasOutput
        for i in range(0,self.numOutput):
            f.write(str(self.biasOutput[0][i])+'\n')

        f.close()

    def loadModel(self,filename):

        f = open(filename,'r')

        #read weightInputHidden
        for i in range(0,self.numInput):
            for j in range(0,self.numHidden):
                self.weightInputHidden[i][j] = string.atof(f.readline()[:-1])

        #read weightHiddenOutput
        for i in range(0,self.numHidden):
            for j in range(0,self.numOutput):
                self.weightHiddenOutput[i][j] = string.atof(f.readline()[:-1])

        #read biasHidden
        for i in range(0,self.numHidden):
            self.biasHidden[0][i] = string.atof(f.readline()[:-1])

        #read biasOutput
        for i in range(0,self.numOutput):
            self.biasOutput[0][i] = string.atof(f.readline()[:-1])

        f.close()
