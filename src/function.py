#! /usr/env/bin
#-*-coding:utf8-*-

### package name : func ###

import math

def sigmoid(x):
    return 1/(1+math.exp(-x))

def diff_sigmoid(x):
    return (1-1/(1+math.exp(-x)))*(1/(1+math.exp(-x)))

def diff_tanh(x):
    return 1-x**2
