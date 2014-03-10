#!/usr/bin/python2.7
# -*- coding=utf8 -*-

import sys
import os
from glob import iglob
import cPickle, gzip
import multiprocessing

import Image
import numpy as np
from scipy.signal import convolve2d

def rgb2yuv(im):
    im = im.convert('YCbCr')
    y, u, v = im.split()
    y = np.asarray(y.getdata()).reshape((im.size[1], im.size[0]))
    u = np.asarray(u.getdata()).reshape((im.size[1], im.size[0]))
    v = np.asarray(v.getdata()).reshape((im.size[1], im.size[0]))
    return y, u, v

def local_normalize(channel):
    k = np.ones((15, 15)) / 225

    diff = channel - convolve2d(channel, k, 'same')
    vari = diff ** 2
    average_vari = np.sqrt(convolve2d(vari, k, 'same'))
    diff /= (average_vari + 0.00000001)
    diff = np.asarray(diff, dtype=np.float32)
    return diff
    
def preProcessing(image):
    y, u, v = rgb2yuv(image)
    pool = multiprocessing.Pool(3)
    y, u, v = pool.map(local_normalize, [y, u, v])
#    y = local_normalize(y)
#    u = local_normalize(u)
#    v = local_normalize(v)
    pool.close()
    pool.join()
    uv = np.concatenate((u.reshape(1, -1), v.reshape(1, -1)), axis=0)
    return [y, uv]

def label(image):
    im = image.resize((60, 80), Image.ANTIALIAS)
    y = np.asarray(im.getdata()).reshape((im.size[1], im.size[0]))
    ones = y < 150
    y = np.zeros((80, 60), dtype=np.int8)
    y[ones] = 1
    y = y.reshape(-1)
    return y

def pkl(path_to_orig, path_to_label, path_to_save):
    os.chdir(path_to_orig)
    pics_orig = iglob('*.png')
    for pic in pics_orig:
        im  = Image.open(pic)
        pic, ext = os.path.splitext(pic)
        y, uv = preProcessing(im)
        im = Image.open(path_to_label + pic + '-profile.png')
        labels = label(im)
        f = gzip.GzipFile(path_to_save + pic + '.pkl.gz', 'wb')
        data = [y, uv, labels]
        f.write(cPickle.dumps(data))
        f.close()

if __name__ == '__main__':
    pkl('/home/daniel/baidu/backup/data/backup/resized-orig/',
        '/home/daniel/baidu/backup/data/backup/resized-train-profiles/',
        '/home/daniel/baidu/data/')
