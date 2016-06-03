#! /usr/env/bin
#-*-coding:utf8-*-

import random

class BPANN(object):

    def __init__(self, ni, nh, no, funcNo):

        #initialize the number of nodes
        self.numInput = ni
        self.numHidden = nh
        self.numOutput = no

        #initialize the weight matrices
        self.weightInputHidden = [[random.uniform(0,0.2) for x in range(1,nh+1)] for y in range(1,ni+1)]
        self.weightHiddenOutput = [[random.uniform(0,4)-2 for x in range(1,no+1)] for y in range(1,nh+1)]

        #initialize the outputs of each layer, as a0, a1, a2
        self.vecInput =
        self.vecHidden =
        self.vecOutput = 


    def feedForward(self):


    def backPropagate(self):
