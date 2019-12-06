# -*- coding: utf-8 -*-

"""Padding operations for cylindrical data.

@@wrap_pad
@@wrap
"""

from __future__ import division

import tensorflow as tf
from math import floor, ceil


def wrap_pad(tensor, wrap_padding, axis=(1, 2)):
    """Apply cylindrical wrapping to one axis and zero padding to another.

    By default, this wraps horizontally and pads vertically. The axes can be
    set with the `axis` keyword, and the wrapping/padding amount can be set
    with the `wrap_pad` keyword.
    """
    rank = tensor.shape.ndims
    if axis[0] >= rank or axis[1] >= rank:
        raise ValueError(
                "Invalid axis for rank-{} tensor (axis={})".format(rank, axis)
              )

    # handle single-number wrap/pad input
    if isinstance(wrap_padding, list) or isinstance(wrap_padding, tuple):
        wrapping = wrap_padding[1]
        padding = wrap_padding[0]
    elif isinstance(wrap_padding, int):
        wrapping = padding = wrap_padding

    # set padding dimensions
    paddings = [[0, 0]] * rank
    paddings[axis[0]] = [floor(padding/2), ceil(padding/2)]

    return tf.pad(wrap(tensor, wrapping, axis=axis[1]), paddings, 'CONSTANT')


def wrap(tensor, wrapping, axis=2):
    """Wrap cylindrically, appending evenly to both sides.

    For odd wrapping amounts, the extra column is appended to the [-1] side.
    """
    rank = tensor.shape.ndims
    if axis >= rank:
        raise ValueError(
                "Invalid axis for rank-{} tensor (axis={})".format(rank, axis)
              )

    sizes = [-1] * rank

    sizes[axis] = ceil(wrapping/2)
    rstarts = [0]*rank
    rpad = tf.slice(tensor, rstarts, sizes)

    sizes[axis] = floor(wrapping/2)
    lstarts = [0]*rank
    lstarts[axis] = tensor.shape.as_list()[axis] - floor(wrapping/2)
    lpad = tf.slice(tensor, lstarts, sizes)

    return tf.concat([lpad, tensor, rpad], axis=axis)

def unwrap(tensor, wrapping, axis=2):
    """Removes wrapping from an image.

    For odd wrapping amounts, this assumes an extra column on the [-1] side.
    """
    rank = tensor.shape.ndims
    if axis >= rank:
        raise ValueError(
                "Invalid axis for rank-{} tensor (axis={})".format(rank, axis)
              )

    sizes = [-1] * rank
    sizes[axis] = tensor.shape.as_list()[axis] - wrapping

    starts = [0] * rank
    starts[axis] = floor(wrapping/2)

    return tf.slice(tensor, starts, sizes)
    #return tensor[:,:,1:-1,:]
