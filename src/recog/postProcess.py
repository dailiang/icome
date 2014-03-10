#!/usr/bin/python2.7
# -*- coding=utf8 -*-

import numpy as np
from scipy.signal import convolve2d as conv2d

import Image, ImageOps

def denoise(labels):
    labels.shape = (80, 60)
    k1 = np.asarray([[1, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1]], dtype=np.uint8)

    k3 = np.asarray([[1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 1],
                     [1, 0, 0, 0, 1],
                     [1, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1]], dtype=np.uint8)

    k5 = np.asarray([[1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)

    k7 = np.asarray([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)

    k9 = np.asarray([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)

    k11 = np.asarray([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)

    k13 = np.asarray([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                     dtype=np.uint8)

    c_k1 = conv2d(labels, k1, mode='same', fillvalue=1)
    highs = c_k1 == 8
    labels[highs] = 1

    c_k3 = conv2d(labels, k3, mode='same', fillvalue=1)
    for i in range(1, 79):
        for j in range(1, 59):
            if c_k3[i][j] == 16:
                labels[i-1:i+2, j-1:j+2] = 1

    c_k5 = conv2d(labels, k5, mode='same', fillvalue=1)
    for i in range(2, 78):
        for j in range(2, 58):
            if c_k5[i][j] == 24:
                labels[i-2:i+3, j-2:j+3] = 1

    c_k7 = conv2d(labels, k7, mode='same', fillvalue=1)
    for i in range(3, 77):
        for j in range(3, 57):
            if c_k7[i][j] == 32:
                labels[i-3:i+4, j-3:j+4] = 1

    c_k9 = conv2d(labels, k9, mode='same', fillvalue=1)
    for i in range(4, 76):
        for j in range(4, 56):
            if c_k9[i][j] == 40:
                labels[i-4:i+5, j-4:j+5] = 1

    c_k11 = conv2d(labels, k11, mode='same', fillvalue=1)
    for i in range(5, 75):
        for j in range(5, 55):
            if c_k11[i][j] == 48:
                labels[i-5:i+6, j-5:j+6] = 1

    c_k13 = conv2d(labels, k13, mode='same', fillvalue=1)
    for i in range(6, 74):
        for j in range(6, 54):
            if c_k13[i][j] == 56:
                labels[i-6:i+7, j-6:j+7] = 1
    # upper border
    for i in range(0, 6):
        for j in range(6, 54):
            if c_k13[i][j] == 56:
                labels[:i+7, j-6:j+7] = 1
    # bottom border
    for i in range(74, 80):
        for j in range(6, 54):
            if c_k13[i][j] == 56:
                labels[i-6:, j-6:j+7] = 1
    # left border
    for i in range(6, 74):
        for j in range(0, 6):
            if c_k13[i][j] == 56:
                labels[i-6:i+7, :j+7] = 1
    # right border
    for i in range(6, 74):
        for j in range(54, 60):
            if c_k13[i][j] == 56:
                labels[i-6:i+7, j-6:] = 1

    return labels


def postProcessing(pic_name, imsize, labels, profile_save_path):
    labels = denoise(labels)
    labels = 1 - labels
    labels = denoise(labels)
    labels = 1 - labels

    h = imsize[1]
    w = int(h * 0.75)
    im = Image.fromarray(np.uint8(labels * 255).reshape(80, 60))
    im = im.convert('RGB')
    im = im.resize((w, h), Image.ANTIALIAS)
    background = Image.new('L', imsize, 'white')

    left_border = (w - imsize[0] + 1) / 2
    if left_border > 0:
        im = ImageOps.crop(im, border=(left_border, 0))
        background.paste(im, (0, 0))
    else:
        background.paste(im, (-left_border, 0))
    background.save(profile_save_path + pic_name + '-profile.jpg')
