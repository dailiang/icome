#!/usr/bin/python2.7
# -*- coding=utf8 -*-

import cPickle
import gzip
import os
import sys
import time
from glob import iglob

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                              dtype=theano.config.floatX),
                               name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,),
                                              dtype=theano.config.floatX),
                               name='b', borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        nll = -T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
        return (T.sum(nll) + T.dot(y, nll) * 2.2) / (1.5 * y.shape[0])

    def accuracy(self, y):
        if y.dtype.startswith('int'):
            return T.sum(T.and_(self.y_pred, y)) * 1. / \
                    T.sum(T.or_(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, 
            activation=T.tanh):
        self.input = input
        if W is None:
            W_values = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                else activation(lin_output))

        self.params = [self.W, self.b]


class ConvPoolTanh(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), W=None, b=None):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = W
        self.b = b

        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape,
                               border_mode='full')
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                          ds=poolsize, ignore_border=True)
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

class Conv(object):
    def __init__(self, rng, input, filter_shape, image_shape, W):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = W

        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape,
                               border_mode='valid')
        self.output = conv_out

        self.params = [self.W]


def evaluate(learning_rate=0.1, 
             train_path='/home/daniel/baidu/data/train/',
             valid_path='/home/daniel/baidu/data/valid/',
             save_path='/home/daniel/baidu/params/',
             params_path='/home/daniel/baidu/params/2.pkl'):
    rng = np.random.RandomState(23455)
    Y = T.matrix('Y')
    UV = T.matrix('UV')
    LABEL = T.ivector('LABEL')

    ###################
    # LOAD THE PARAMS #
    ###################
    fle = open(params_path)
    
    # layer0
    W_y = theano.shared(value=cPickle.load(fle), borrow=True)
    b_y = theano.shared(value=cPickle.load(fle), borrow=True)
    W_uv = theano.shared(value=cPickle.load(fle), borrow=True)
    b_uv = theano.shared(value=cPickle.load(fle), borrow=True)
    W_1 = theano.shared(value=cPickle.load(fle), borrow=True)
    b_1 = theano.shared(value=cPickle.load(fle), borrow=True)
    W_2 = theano.shared(value=cPickle.load(fle), borrow=True)
    fle.close()

    
    #######################
    # BUILD ACTURAL MODEL #
    #######################

    print '... start building the actual model'

    # layer 0
    layer0_y_input = Y.reshape((1, 1, 320, 240))
    layer0_y = ConvPoolTanh(rng, input=layer0_y_input,
                image_shape=(1, 1, 320, 240),
                filter_shape=(10, 1, 7, 7), W=W_y, b=b_y)

    layer0_uv_input = UV.reshape((1, 2, 320, 240))
    layer0_uv = ConvPoolTanh(rng, input=layer0_uv_input,
                image_shape=(1, 2, 320, 240),
                filter_shape=(6, 2, 7, 7), W=W_uv, b=b_uv)

    # concatenate
    layer1_input = T.concatenate((layer0_y.output, layer0_uv.output), axis=1)
    layer1 = ConvPoolTanh(rng, input=layer1_input,
                image_shape=(1, 16, 163, 123),
                filter_shape=(64, 16, 6, 6), W=W_1, b=b_1)

    layer2 = Conv(rng, input=layer1.output,
                image_shape=(1, 64, 84, 64),
                  filter_shape=(256, 64, 5, 5), W=W_2)

    layer3_input = layer2.output.reshape((256, 4800)).dimshuffle(1, 0)

    layer3 = HiddenLayer(rng, input=layer3_input, n_in=256, n_out=500,
            activation=T.tanh)

    layer4 = LogisticRegression(input=layer3.output, n_in=500, n_out=2)

    cost = layer4.negative_log_likelihood(LABEL)

    valid_model = theano.function([Y, UV, LABEL], layer4.accuracy(LABEL))

    params = layer4.params + layer3.params

    grads = T.grad(cost, params)
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([Y, UV, LABEL], cost, updates=updates)

    
    ##################
    # START TO TRAIN #
    ##################
    epoch = 0
    print '... training started'

    while 1:
        # train phase
        epoch += 1
        os.chdir(train_path)
        dataset = iglob('*.gz')
        print '... train phase'
        print time.localtime()
        for data in dataset:
            f = gzip.open(data, 'rb')
            y, uv, label = cPickle.load(f)
            f.close()
            train_model(y, uv, label)

        print '... save params'
        save_file = open(save_path + 'shared-mlp-' + str(epoch) + '.pkl', 'wb')
        cPickle.dump(params, save_file, -1)
        save_file.close()

        save_file = open(save_path + 'mlp' + str(epoch) + '.pkl', 'wb')
        cPickle.dump(layer0_y.W.get_value(borrow=True), save_file, -1)
        cPickle.dump(layer0_y.b.get_value(borrow=True), save_file, -1)
        cPickle.dump(layer0_uv.W.get_value(borrow=True), save_file, -1)
        cPickle.dump(layer0_uv.b.get_value(borrow=True), save_file, -1)
        cPickle.dump(layer1.W.get_value(borrow=True), save_file, -1)
        cPickle.dump(layer1.b.get_value(borrow=True), save_file, -1)
        cPickle.dump(layer2.W.get_value(borrow=True), save_file, -1)
        cPickle.dump(layer3.W.get_value(borrow=True), save_file, -1)
        cPickle.dump(layer3.b.get_value(borrow=True), save_file, -1)
        cPickle.dump(layer4.W.get_value(borrow=True), save_file, -1)
        cPickle.dump(layer4.b.get_value(borrow=True), save_file, -1)
        save_file.close()

        # valid after every epoch
        os.chdir(valid_path)
        dataset = iglob('*.gz')
        print '... valid phase'
        print time.localtime()
        valid_accuracy = []
        for data in dataset:
            f = gzip.open(data, 'rb')
            y, uv, label = cPickle.load(f)
            f.close()
            accuracy = valid_model(y, uv, label)
            valid_accuracy.append(accuracy)
        this_valid_accuracy = np.mean(valid_accuracy)
        print('epoch %i, validation accuracy %f %%\n\n' %
              (epoch, this_valid_accuracy * 100))

        # valid train dataset after each 10 epochs
        if epoch % 10 == 0:
            os.chdir(train_path)
            dataset = iglob('*.gz')
            print '... valid train accuracy'
            valid_accuracy = []
            for data in dataset:
                f = gzip.open(data, 'rb')
                y, uv, label = cPickle.load(f)
                f.close()
                accuracy = valid_model(y, uv, label)
                valid_accuracy.append(accuracy)
            this_valid_accuracy = np.mean(valid_accuracy)
            print('epoch %i, validation accuracy %f %%\n\n' %
                  (epoch, this_valid_accuracy * 100))

if __name__ == '__main__':
    theano.config.floatX = 'float32'
    evaluate()
