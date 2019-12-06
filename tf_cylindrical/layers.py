# -*- coding: utf-8 -*-

"""Convolutional layers for cylindrical data.

@@convolution2d
@@conv2d
"""

from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tf_cylindrical.pad import wrap, wrap_pad


def convolution2d(inputs, num_outputs, kernel_size, stride=1, padding='CYLIN', *args, **kwargs):
    # kernel size 1D -> 2D
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]

    # maintain original behavior
    print(padding)
    if padding=='SAME' or padding=='VALID':
        return slim.conv2d(inputs,
                           num_outputs,
                           kernel_size,
                           stride,
                           padding,
                           *args,
                           **kwargs)

    # W=(W−F+2P)/S+1
    elif padding=='CYLIN':
        size = inputs.get_shape()
        height = size[1]
        width = size[2]
        wrap_padding = [k-1 for k in kernel_size]
        wrapped_inputs = wrap_pad(inputs, wrap_padding)

        return slim.conv2d(wrapped_inputs,
                           num_outputs,
                           kernel_size,
                           stride,
                           'VALID',
                           *args,
                           **kwargs)
    raise('Not a valid padding: {}'.format(padding))


# Aliases
conv2d = convolution2d
