#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

__author__ = 'Yasuhiro'
__date__ = '2018/3/21'


class WhiteEffectCountLayer(snt.AbstractModule):
    def __init__(self, name='white_effect_count'):
        super().__init__(name=name)

    def _build(self, short_effect, long_effect, ou_effect):
        """
        非手番側の利きがマスごとにいくつあるかを計算する

        :param short_effect:
        :param long_effect:
        :param ou_effect:
        :return:
        """
        white_effect = [tf.to_int32(effect)
                        for effect in short_effect + long_effect + ou_effect]
        # マスごとの利きの個数を計算
        count = tf.add_n(white_effect)

        return count
