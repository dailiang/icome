#!/usr/bin/python2.7
# -*- coding=utf8 -*-

import os
import sys
import cPickle
from glob import glob
import Image, ImageOps

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from preProcess import preProcessing
from postProcess import postProcessing

class CnnMlp(object):
    '''5-layers model:
    layer 0: Conv + pool + tanh
    layer 1: Conv + pool + tanh
    layer 2: Conv
    layer 3: Fully-connected Hidden layer with Tanh activation function
    layer 4: Logistic regression
    '''
    def __init__(self, y, uv, params):
        self.layer0_W_y, self.layer0_b_y = params[0]
        self.layer0_W_uv, self.layer0_b_uv = params[1]
        self.layer1_W, self.layer1_b = params[2]
        self.layer2_W = params[3]
        self.layer3_W, self.layer3_b = params[4]
        self.layer4_W, self.layer4_b = params[5]

        poolsize = (2, 2)
        # layer0_y: conv-maxpooling-tanh
        layer0_y_conv = conv.conv2d(input=y, filters=self.layer0_W_y,
                border_mode='full')
        layer0_y_pool = downsample.max_pool_2d(input=layer0_y_conv,
                ds=poolsize, ignore_border=True)
        layer0_y_out = T.tanh(layer0_y_pool + \
                self.layer0_b_y.reshape(1, -1, 1, 1))

        # layer0_uv: conv-maxpooling-tanh
        layer0_uv_conv = conv.conv2d(input=uv, filters=self.layer0_W_uv,
                border_mode='full')
        layer0_uv_pool = downsample.max_pool_2d(input=layer0_uv_conv,
                ds=poolsize, ignore_border=True)
        layer0_uv_out = T.tanh(layer0_uv_pool + \
                self.layer0_b_uv.reshape(1, -1, 1, 1))

        layer1_input = T.concatenate((layer0_y_out, layer0_uv_out), axis=1)

        # layer1: conv-maxpooling-tanh
        layer1_conv = conv.conv2d(input=layer1_input, filters=self.layer1_W,
                border_mode='full')
        layer1_pool = downsample.max_pool_2d(input=layer1_conv,
                ds=poolsize, ignore_border=True)
        layer1_out = T.tanh(layer1_pool + self.layer1_b.reshape(1, -1, 1, 1))

        # layer2: conv
        layer2_out = conv.conv2d(input=layer1_out, filters=self.layer2_W,
                border_mode='valid')

        layer3_input = layer2_out.reshape((256, -1)).dimshuffle(1, 0)

        # layer3: hidden-layer
        layer3_lin = T.dot(layer3_input, self.layer3_W) + self.layer3_b
        layer3_out = T.tanh(layer3_lin)

        # layer4: logistic-regression
        layer4_out = T.nnet.softmax(T.dot(layer3_out, self.layer4_W) + \
                self.layer4_b)
        self.pred = T.argmin(layer4_out, axis=1)

def loadParams(params_path='model_params.pkl'):
    '''load model's parameters
    '''
    fle = open(params_path)
    print '... loading the parameters'

    W_y = np.float64(cPickle.load(fle))
    b_y = np.float64(cPickle.load(fle))
    W_uv = np.float64(cPickle.load(fle))
    b_uv = np.float64(cPickle.load(fle))
    W_1 = np.float64(cPickle.load(fle))
    b_1 = np.float64(cPickle.load(fle))
    W_2 = np.float64(cPickle.load(fle))
    W_3 = np.float64(cPickle.load(fle))
    b_3 = np.float64(cPickle.load(fle))
    W_4 = np.float64(cPickle.load(fle))
    b_4 = np.float64(cPickle.load(fle))
    fle.close()
    
    return [(W_y, b_y), (W_uv, b_uv), (W_1, b_1), (W_2), (W_3, b_3), (W_4, b_4)] 


def main(origin_pics_path, profile_save_path):
    ##################
    # BULIDING MODEL #
    ##################
    print '\n... process0 started'
    print '... building model'
    Y = T.matrix('Y')
    UV = T.matrix('UV')
    params = loadParams()

    Y_input = Y.reshape((1, 1, 320, 240))
    UV_input = UV.reshape((1, 2, 320, 240))
    CNNMLP = CnnMlp(Y_input, UV_input, params)
    classifier = theano.function([Y, UV], CNNMLP.pred)

    ##################
    # CLASSIFY START #
    ##################
    os.chdir(origin_pics_path)
    pics = glob('*')
    for pic in pics[: len(pics) / 4]:
        pic_name, ext = os.path.splitext(pic)
        im = Image.open(pic)
        y, uv = preProcessing(im)
        labels = classifier(y, uv)
        postProcessing(pic_name, im.size, labels, profile_save_path)

if __name__ == '__main__':
    theano.config.floatX = 'float64'
    if len(sys.argv) != 3:
        print 'Invalid number of arguments passed.'
    else:
        origin_pics_path = sys.argv[1]
        profile_save_path = sys.argv[2]
        if not os.path.exists(profile_save_path):
            os.mkdir(profile_save_path)
        main(origin_pics_path, profile_save_path)
        print 'process0 finished.'
