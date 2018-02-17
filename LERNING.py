# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:26:10 2017

@author: qoliver
"""


''' Input=>Weighting=>Layer 1(activation func)=>weighting=>Layer 2 => weighting=> Output

Putpit=> Eval by cost func => Optimazation=> HOORAY'''
import tensorflow as tf
import numpy as np



#Training
    
l1Neurons = 100
nClasses = 0
batch = 100
def neural_network_model(data):
    data=np.float32(data)
    h1Layer = {'weights':tf.Variable(tf.random_normal([l1Neurons, nClasses])),
                'biases':tf.Variable(tf.random_normal([nClasses]))}
    output= tf.matmul(data, h1Layer['weights'])+h1Layer['biases']
    return output




