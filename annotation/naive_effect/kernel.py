#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
各方向の聴きを求めるための畳み込みフィルタを作る
フィルタのサイズは[filter_height, filter_width, in_channels, out_channels]

縦型で相手側から手番側の順序のマスの並びになっている

目的地の駒に関係なく効きはある
あくまで求めるのは、n-1マス利きがあるか(移動できるか)どうかなので、途中に駒があると利きはない
"""

from collections import namedtuple

import numpy as np
import tensorflow as tf

__author__ = 'Yasuhiro'
__date__ = '2018/1/27'


Index = namedtuple('Index', ['source', 'target'])


def make_long_kernel(direction, size):
    name = 'long_kernel_{}{}'.format(direction.name, size)
    collection = tf.get_collection_ref(name)
    if len(collection) == 0:
        # まだない
        f_list = [make_long_right_up_kernel, make_long_right_kernel,
                  make_long_right_down_kernel, make_long_up_kernel,
                  make_long_down_kernel, make_long_left_up_kernel,
                  make_long_left_kernel, make_long_left_down_kernel]

        kernel = tf.constant(f_list[direction.value](size=size),
                             dtype=tf.float32, name=name)

        tf.add_to_collection(name, kernel)
    else:
        kernel = collection[0]

    return kernel


def make_short_kernel(direction):
    name = 'short_kernel_{}'.format(direction.name)
    collection = tf.get_collection_ref(name)
    if len(collection) == 0:
        # まだない
        f_list = [make_short_right_up_kernel, make_short_right_kernel,
                  make_short_right_down_kernel, make_short_up_kernel,
                  make_short_down_kernel, make_short_left_up_kernel,
                  make_short_left_kernel, make_short_left_down_kernel,
                  # 桂馬の動き
                  make_short_right_up_up_kernel, make_short_left_up_up_kernel,
                  make_short_right_down_down_kernel,
                  make_short_left_down_down_kernel]

        kernel = tf.constant(f_list[direction.value](), dtype=tf.float32,
                             name=name)

        tf.add_to_collection(name, kernel)
    else:
        kernel = collection[0]

    return kernel


def get_up_index():
    return Index(source=-1, target=0)


def get_down_index():
    return Index(source=0, target=-1)


def get_right_index():
    return Index(source=-1, target=0)


def get_left_index():
    return Index(source=0, target=-1)


def make_long_vertical_kernel(size, w):
    shape = (1, size, 2, 1)
    kernel = np.empty(shape)

    # target
    kernel[0, w.target, :, 0] = (0, 0)
    # source
    kernel[0, w.source, :, 0] = (0, 1)
    # midway
    kernel[0, 1:-1] = -1

    return kernel


def make_short_vertical_kernel(w):
    shape = (1, 2, 1, 1)
    kernel = np.empty(shape)

    # target
    kernel[0, w.target] = 0
    # source
    kernel[0, w.source] = 1

    return kernel


def make_long_horizontal_kernel(size, h):
    shape = (size, 1, 2, 1)
    kernel = np.empty(shape)

    # target
    kernel[h.target, 0, :, 0] = (0, 0)
    # source
    kernel[h.source, 0, :, 0] = (0, 1)
    # midway
    kernel[1:-1, 0] = -1

    return kernel


def make_short_horizontal_kernel(h):
    shape = (2, 1, 1, 1)
    kernel = np.empty(shape)

    # target
    kernel[h.target] = 0
    # source
    kernel[h.source] = 1

    return kernel


def make_long_diagonal1_kernel(size, h, w):
    """
    初期配置の角交換の方向の利きを求める

    :param size:
    :param h:
    :param w:
    :return:
    """
    shape = (size, size, 2, 1)
    kernel = np.zeros(shape)

    # target
    kernel[h.target, w.target, :, 0] = (0, 0)
    # source
    kernel[h.source, w.source, :, 0] = (0, 1)
    # midway
    for i in range(1, size - 1):
        kernel[i, i] = -1

    return kernel


def make_short_diagonal1_kernel(h, w):
    shape = (2, 2, 1, 1)
    kernel = np.zeros(shape)

    # target
    kernel[h.target, w.target] = 0
    # source
    kernel[h.source, w.source] = 1

    return kernel


def make_long_diagonal2_kernel(size, h, w):
    """
    diagonal1と直交する方向の利きを求める

    :param size:
    :param h:
    :param w:
    :return:
    """
    shape = (size, size, 2, 1)
    kernel = np.zeros(shape)

    # target
    kernel[h.target, w.target, :, 0] = (0, 0)
    # source
    kernel[h.source, w.source, :, 0] = (0, 1)
    # midway
    for i in range(1, size - 1):
        kernel[i, size - i - 1] = -1

    return kernel


def make_short_diagonal2_kernel(h, w):
    shape = (2, 2, 1, 1)
    kernel = np.zeros(shape)

    # target
    kernel[h.target, w.target] = 0
    # source
    kernel[h.source, w.source] = 1

    return kernel


def make_long_up_kernel(size):
    """
    手番側から見て相手方向へ進む利きを求める

    :param size:
    :return:
    """
    w = get_up_index()
    return make_long_vertical_kernel(size=size, w=w)


def make_short_up_kernel():
    return make_short_vertical_kernel(w=get_up_index())


def make_long_down_kernel(size):
    """
    手番側から見て自身の方向へ進む利きを求める

    :param size:
    :return:
    """
    w = get_down_index()
    return make_long_vertical_kernel(size=size, w=w)


def make_short_down_kernel():
    return make_short_vertical_kernel(w=get_down_index())


def make_long_right_kernel(size):
    """
    手番側から見て右方向へ動く利きを求める

    :param size:
    :return:
    """
    h = get_right_index()
    return make_long_horizontal_kernel(size=size, h=h)


def make_short_right_kernel():
    return make_short_horizontal_kernel(h=get_right_index())


def make_long_left_kernel(size):
    """
    手番側から見て左方向へ進む利きを求める
    :param size:
    :return:
    """
    h = get_left_index()
    return make_long_horizontal_kernel(size=size, h=h)


def make_short_left_kernel():
    return make_short_horizontal_kernel(h=get_left_index())


def make_long_right_up_kernel(size):
    """
    手番側から見て右上方向へ進む利きを求める
    :param size:
    :return:
    """
    h = get_right_index()
    w = get_up_index()
    return make_long_diagonal1_kernel(size=size, h=h, w=w)


def make_short_right_up_kernel():
    return make_short_diagonal1_kernel(h=get_right_index(), w=get_up_index())


def make_long_left_down_kernel(size):
    """
    手番側から見て左下方向へ進む利きを求める
    :param size:
    :return:
    """
    h = get_left_index()
    w = get_down_index()
    return make_long_diagonal1_kernel(size=size, h=h, w=w)


def make_short_left_down_kernel():
    return make_short_diagonal1_kernel(h=get_left_index(), w=get_down_index())


def make_long_left_up_kernel(size):
    """
    手番側から見て左上方向へ進む利きを求める
    :param size:
    :return:
    """
    h = get_left_index()
    w = get_up_index()
    return make_long_diagonal2_kernel(size=size, h=h, w=w)


def make_short_left_up_kernel():
    return make_short_diagonal1_kernel(h=get_left_index(), w=get_up_index())


def make_long_right_down_kernel(size):
    """
    手番側から見て右下方向へ進む利きを求める
    :param size:
    :return:
    """
    h = get_right_index()
    w = get_down_index()
    return make_long_diagonal2_kernel(size=size, h=h, w=w)


def make_short_right_down_kernel():
    return make_short_diagonal1_kernel(h=get_right_index(), w=get_down_index())


def make_short_right_up_up_kernel():
    shape = (2, 3, 1, 1)
    kernel = np.zeros(shape)

    # target
    kernel[0, 0] = 0
    # source
    kernel[1, 2] = 1

    return kernel


def make_short_left_up_up_kernel():
    shape = (2, 3, 1, 1)
    kernel = np.zeros(shape)

    # target
    kernel[1, 0] = 0
    # source
    kernel[0, 2] = 1

    return kernel


def make_short_right_down_down_kernel():
    shape = (2, 3, 1, 1)
    kernel = np.zeros(shape)

    # target
    kernel[0, 2] = 0
    # source
    kernel[1, 0] = 1

    return kernel


def make_short_left_down_down_kernel():
    shape = (2, 3, 1, 1)
    kernel = np.zeros(shape)

    # target
    kernel[1, 2] = 0
    # source
    kernel[0, 0] = 1

    return kernel
