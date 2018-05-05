#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
畳み込み演算での利きの判定結果に対してパディングを行う

畳み込みはpadding='valid'で行われた
"""

import tensorflow as tf

from annotation.direction import Direction

__author__ = 'Yasuhiro'
__date__ = '2018/1/29'


def pad(flag, direction, data_format, kernel_size=None):
    functions = [pad_right_up, pad_right, pad_right_down,
                 pad_up, pad_down,
                 pad_left_up, pad_left, pad_left_down,
                 # 桂馬
                 pad_right_up_up, pad_left_up_up,
                 pad_right_down_down, pad_left_down_down]
    f_pad = functions[direction.value]
    if direction.value >= Direction.RIGHT_UP_UP.value:
        return f_pad(flag=flag, data_format=data_format)
    else:
        return f_pad(flag=flag, kernel_size=kernel_size,
                     data_format=data_format)


def pad_helper(flag, h, w, data_format):
    if data_format == 'NCHW':
        paddings = tf.constant([[0, 0], [0, 0], h, w], dtype=tf.int32)
    else:
        paddings = tf.constant([[0, 0], h, w, [0, 0]], dtype=tf.int32)
    result = tf.pad(flag, paddings)
    return result


def pad_up(flag, kernel_size, data_format):
    return pad_helper(flag=flag, h=[0, 0], w=[0, kernel_size - 1],
                      data_format=data_format)


def pad_down(flag, kernel_size, data_format):
    return pad_helper(flag=flag, h=[0, 0], w=[kernel_size - 1, 0],
                      data_format=data_format)


def pad_right(flag, kernel_size, data_format):
    return pad_helper(flag=flag, h=[0, kernel_size - 1], w=[0, 0],
                      data_format=data_format)


def pad_left(flag, kernel_size, data_format):
    return pad_helper(flag=flag, h=[kernel_size - 1, 0], w=[0, 0],
                      data_format=data_format)


def pad_right_up(flag, kernel_size, data_format):
    return pad_helper(flag=flag, h=[0, kernel_size - 1],
                      w=[0, kernel_size - 1], data_format=data_format)


def pad_left_down(flag, kernel_size, data_format):
    return pad_helper(flag=flag, h=[kernel_size - 1, 0],
                      w=[kernel_size - 1, 0], data_format=data_format)


def pad_right_down(flag, kernel_size, data_format):
    return pad_helper(flag=flag, h=[0, kernel_size - 1],
                      w=[kernel_size - 1, 0], data_format=data_format)


def pad_left_up(flag, kernel_size, data_format):
    return pad_helper(flag=flag, h=[kernel_size - 1, 0],
                      w=[0, kernel_size - 1], data_format=data_format)


def pad_right_up_up(flag, data_format):
    return pad_helper(flag=flag, h=[0, 1], w=[0, 2], data_format=data_format)


def pad_left_up_up(flag, data_format):
    return pad_helper(flag=flag, h=[1, 0], w=[0, 2], data_format=data_format)


def pad_right_down_down(flag, data_format):
    return pad_helper(flag=flag, h=[0, 1], w=[2, 0], data_format=data_format)


def pad_left_down_down(flag, data_format):
    return pad_helper(flag=flag, h=[1, 0], w=[2, 0], data_format=data_format)
