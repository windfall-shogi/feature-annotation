#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

__author__ = 'Yasuhiro'
__date__ = '2018/2/17'


def get_axis(data_format):
    return 1 if data_format == 'NCHW' else -1


def make_one_hot(index_board, data_format):
    one_hot = tf.one_hot(index_board, 2,
                         axis=get_axis(data_format=data_format))
    if data_format == 'NCHW':
        shape = [-1, 2, 9, 9]
    else:
        shape = [1, 9, 9, 2]
    one_hot = tf.reshape(one_hot, shape)

    return one_hot
