#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import sonnet as snt

__author__ = 'Yasuhiro'
__date__ = '2018/2/17'


class CombineMovableMaskLayer(snt.AbstractModule):
    def __init__(self, name='combine_movable_mask'):
        super().__init__(name=name)

    def _build(self, effects):
        """
        長さ10の配列を一つにまとめる
        王手されている場合に動ける場所のマスクを計算する
        王手の場合はそのマスにしか動けないので、andの計算で絞り込む
        2方向から王手されている可能性もあるので、その場合は全てFalseになる

        :param effects:
        :return:
        """
        effect = merge(*effects)
        return effect


def merge(*args):
    n = len(args)
    if n == 1:
        return args[0]
    elif n == 2:
        return tf.logical_and(*args)

    tmp = [tf.logical_and(v1, v2) for v1, v2 in zip(args[0::2], args[1::2])]
    if n % 2 == 1:
        tmp.append(args[-1])
    return merge(*tmp)
