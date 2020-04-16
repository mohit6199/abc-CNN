import numpy as np
from copy import deepcopy

import keras.backend
from keras.models import Model, Sequential
from keras.layers import Input, Add, Dense, Dropout, Flatten
from keras.layers import Activation, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.optimizers import Adam, Nadam
from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

import os
import tensorflow as tf
from copy import deepcopy

try:
    # Python 2 module
    from itertools import izip_longest as zip_longest
except ImportError:
    # Python 3 module
    from itertools import zip_longest

def add_conv(layers, max_out_ch, conv_kernel):
    out_channel = np.random.randint(3, max_out_ch)
    conv_kernel = np.random.randint(3, conv_kernel)

    layers.append({"type": "conv", "ou_c": out_channel, "kernel": conv_kernel})

    return layers

def add_res(layers, max_out_ch, conv_kernel):
    out_channel = np.random.randint(3, max_out_ch)
    conv_kernel = np.random.randint(3, conv_kernel)

    layers.append({"type": "res", "ou_c": out_channel, "kernel": conv_kernel})

    return layers


def add_fc(layers, max_fc_neurons):
    layers.append({"type": "fc", "ou_c": np.random.randint(1, max_fc_neurons), "kernel": -1})
    
    return layers
        

def add_pool(layers, fc_prob, num_pool_layers, max_pool_layers, max_out_ch, max_conv_kernel, max_fc_neurons, output_dim):
    pool_layers = num_pool_layers

    if pool_layers < max_pool_layers:
        random_pool = np.random.rand()
        pool_layers += 1
        if random_pool < 0.5:
            # Add Max Pooling
            layers.append({"type": "max_pool", "ou_c": -1, "kernel": 2})
        else:
            layers.append({"type": "avg_pool", "ou_c": -1, "kernel": 2})
    
    return layers, pool_layers


def differenceConvPool(p1, p2, const):
    diff = []

    for comb in zip_longest(p1, p2):
        if comb[0] != None and comb[1] != None:
            if np.random.uniform() <= const:
                diff.append(comb[0])
            else:
                diff.append(comb[1])

        elif comb[0] != None and comb[1] == None:
            diff.append(comb[0])


    
    return diff


def differenceFC(p1, p2, const):
    diff = []

    # Compute the difference from the end to the begin
    for comb in zip_longest(p1[::-1], p2[::-1]):
        if comb[0] != None and comb[1] != None:
            if np.random.uniform() <= const:
                diff.append(comb[0])
            else:
                diff.append(comb[1])

        elif comb[0] != None and comb[1] == None:
            diff.append(comb[0])


    diff = diff[::-1]
    
    return diff


def computeDifference(p1, p2, const):
    diff = []
    # First, find the index where the fully connected layers start in each particle
    p1fc_idx = next((index for (index, d) in enumerate(p1) if d["type"] == "fc"))
    p2fc_idx = next((index for (index, d) in enumerate(p2) if d["type"] == "fc"))

    # Compute the difference only between the convolution and pooling layers
    diff.extend(differenceConvPool(p1[0:p1fc_idx], p2[0:p2fc_idx],const))
    
    # Compute the difference between the fully connected layers 
    diff.extend(differenceFC(p1[p1fc_idx:], p2[p2fc_idx:],const))
    
    keep_all_layers = True
    for i in range(len(diff)):
        if diff[i]["type"] != "keep" or diff[i]["type"] != "keep_fc":
            keep_all_layers = False
            break

    return diff, keep_all_layers


def generateNewSolution(p1,p2):#p1 is the proper one and the p2 is the random neighbour one
    l1 = len(p1)
    l2 = len(p2)
    new_p = []
    p1_fc_idx = next((index for (index, d) in enumerate(p1) if d["type"] == "fc" or d["type"] == "keep_fc" or d["type"] == "remove_fc"))
    p2_fc_idx = next((index for (index, d) in enumerate(p2) if d["type"] == "fc" or d["type"] == "keep_fc" or d["type"] == "remove_fc"))
    #print("pos of fc_layer = ",p1_fc_idx ,"  ",p2_fc_idx)
    pos_conv = np.random.randint(0,min(p1_fc_idx,p2_fc_idx))
    #print(pos_conv)
    #pos_fc = np.random.randint(min(p1_fc_idx,p2_fc_idx),min(l1,l2))
    while p1[pos_conv]['type'] != 'conv' and p2[pos_conv]['type']!= 'conv':
        pos_conv += 1
        if pos_conv>min(p1_fc_idx,p2_fc_idx):
            pos_conv = 0

    #print("print pos of conv =",pos_conv)
    #print(pos_fc)
    out1 = p1[pos_conv]['ou_c']
    out2 = p2[pos_conv]['ou_c']
    #print("out1 = ",out1)
    #print("out2 = ", out2)
    r = np.random.uniform()
    #print(out1 , "+" , r , "(" , out1 , "-" , out2 , ")")
    new_out =int(out1 + r*(out1-out2))
    #print("new Out = ",new_out)
    if new_out < 3:
        new_out = 3
    for i in range(0,l1):
        #print(i)
        if i != pos_conv:
            new_p.append(p1[i])
        else:
            x = {'type': 'conv', 'ou_c': new_out, 'kernel': p1[i]['kernel']}
            new_p.append(x)

    #p1[pos_conv]['ou_c'] = new_out
    #print(new_p)
    return new_p