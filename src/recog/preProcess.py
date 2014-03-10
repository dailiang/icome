#!/usr/bin/python2.7
# -*- coding=utf8 -*-

import os
import sys

import Image,ImageOps

import numpy as np
from scipy.signal import convolve2d

def cropResize(im):
    if im.mode == 'RGBA':
        im.load()
        background = Image.new('RGB', im.size, (0, 0, 0))
        background.paste(im, im.split()[-1])
        im = background
    elif im.mode != 'RGB':
        im = im.convert('RGB')

    left_border = int(im.size[1] * 0.75 - im.size[0]) / 2
    if left_border > 0:
        try:
            im = ImageOps.expand(im, border=(left_border, 0), fill=(0, 0, 0))
        except:
            im = ImageOps.expand(im, border=(left_border, 0), fill=0)
    else:
        im = ImageOps.crop(im, border=(-left_border, 0))

    im = im.resize((240, 320), Image.ANTIALIAS)
    return im

def rgb2yuv(im):
    #im = im.convert('YCbCr')
    tran = (0.299, 0.587, 0.114, 0,
            -0.169, -0.331, 0.5, 128,
            0.5, -0.419, -0.081, 128)
    im = im.convert('RGB', tran)
    y, u, v = im.split()
    y = np.asarray(y.getdata()).reshape((im.size[1], im.size[0]))
    u = np.asarray(u.getdata()).reshape((im.size[1], im.size[0]))
    v = np.asarray(v.getdata()).reshape((im.size[1], im.size[0]))
    return y, u, v

def local_normalize(channel):
    k = np.ones((15, 15)) / 225.

    diff = channel - convolve2d(channel, k, 'same')
    vari = diff ** 2
    average_vari = np.sqrt(convolve2d(vari, k, 'same'))
    output = np.float64(diff / (average_vari + 0.00000001))
    return output

def preProcessing(im):
    im = cropResize(im)
    y, u, v = rgb2yuv(im)
    y = local_normalize(y)
    u = local_normalize(u)
    v = local_normalize(v)
    uv = np.concatenate((u.reshape(1, -1), v.reshape(1, -1)), axis=0)
    return [y, uv]
