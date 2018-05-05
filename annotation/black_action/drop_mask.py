#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

__author__ = 'Yasuhiro'
__date__ = '2018/2/22'


def make_drop_mask1(data_format):
    """
    FU, KYが成らずにいれる領域のマスクを作成
    FU, KYを打てる領域のマスクでもある

    成らずにいれるならTrue, 成るしかないならばFalse

    :param data_format:
    :return:
    """
    name = 'black_drop_mask1'
    collection = tf.get_collection_ref(name)
    if len(collection):
        return collection[0]

    if data_format == 'NCHW':
        mask = np.ones((1, 1, 9, 9), dtype=np.bool)
        mask[:, :, :, 0] = False
    else:
        mask = np.ones((1, 9, 9, 1), dtype=np.bool)
        mask[:, :, 0, :] = False

    mask = tf.constant(mask, dtype=tf.bool)

    tf.add_to_collection(name, mask)

    return mask


def make_drop_mask2(data_format):
    """
    KEが成らずにいれる領域のマスクを作成
    KEを打てる領域のマスクでもある

    成らずにいれるならTrue, 成るしかないならばFalse

    :param data_format:
    :return:
    """
    name = 'black_drop_mask2'
    collection = tf.get_collection_ref(name)
    if len(collection):
        return collection[0]

    if data_format == 'NCHW':
        mask = np.ones((1, 1, 9, 9), dtype=np.bool)
        mask[:, :, :, :2] = False
    else:
        mask = np.ones((1, 9, 9, 1), dtype=np.bool)
        mask[:, :, :2, :] = False

    mask = tf.constant(mask, dtype=tf.bool)

    tf.add_to_collection(name, mask)

    return mask
