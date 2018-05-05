#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from ..direction import Direction

__author__ = 'Yasuhiro'
__date__ = '2018/2/22'


def make_promotion_mask(step_size, data_format, direction):
    if direction in (Direction.RIGHT_DOWN_DOWN, Direction.LEFT_DOWN_DOWN):
        raise ValueError(direction)

    if direction in (Direction.RIGHT_DOWN, Direction.DOWN,
                     Direction.LEFT_DOWN):
        return make_promotion_mask_down(step_size=step_size,
                                        data_format=data_format)
    else:
        return make_promotion_mask_up(step_size=step_size,
                                      data_format=data_format)


# noinspection PyUnusedLocal
def make_promotion_mask_up(step_size, data_format):
    """
    右上、右、上、左上、左方向に移動した時の成りの領域のマスク
    成れる場所ならばTrue、それ以外はFalse

    :param step_size:
    :param data_format:
    :return:
    """
    name = 'black_promotion_mask_up'
    collection = tf.get_collection_ref(name)
    if len(collection):
        return collection[0]

    if data_format == 'NCHW':
        mask = np.zeros((1, 1, 9, 9), dtype=np.bool)
        mask[:, :, :, :3] = True
    else:
        mask = np.zeros((1, 9, 9, 1), dtype=np.bool)
        mask[:, :, :3, :] = True
    mask = tf.constant(mask, dtype=tf.bool)

    tf.add_to_collection(name, mask)

    return mask


def make_promotion_mask_down(step_size, data_format):
    """
    右下、下、左下方向に移動した時の成りの領域のマスク
    成れる場所ならばTrue、それ以外はFalse

    :param step_size:
    :param data_format:
    :return:
    """
    name = 'black_promotion_mask_down{}'.format(step_size)
    collection = tf.get_collection_ref(name)
    if len(collection):
        return collection[0]

    if data_format == 'NCHW':
        mask = np.zeros((1, 1, 9, 9), dtype=np.bool)
        mask[:, :, :, step_size:step_size + 3] = True
    else:
        mask = np.zeros((1, 9, 9, 1), dtype=np.bool)
        mask[:, :, step_size:step_size + 3, :] = True
    mask = tf.constant(mask, dtype=tf.bool)

    tf.add_to_collection(name, mask)

    return mask
