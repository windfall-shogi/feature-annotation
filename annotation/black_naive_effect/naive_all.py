#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from .naive_short import BlackNaiveShortEffectLayer
from .naive_long import BlackNaiveLongEffectLayer
from ..direction import get_eight_directions, Direction
from ..naive_effect import CombineLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/15'


class BlackNaiveAllEffect(snt.AbstractModule):
    def __init__(self, data_format, use_cudnn=True,
                 name='black_naive_all_effect'):
        super().__init__(name=name)
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, board):
        """
        手番側のナイーブな利きを全て求める

        同時にナイーブな長い利きも返す
        非手番側の駒がピンされているかを判定に利用する

        :param board:
        :return:
        """
        effect_list = []
        long_effects = {}
        for direction in get_eight_directions():
            short_effect = BlackNaiveShortEffectLayer(
                direction=direction, data_format=self.data_format,
                use_cudnn=self.use_cudnn,
                name='black_naive_short_effect_{}'.format(direction.name)
            )(board)
            long_effect = BlackNaiveLongEffectLayer(
                direction=direction, data_format=self.data_format,
                use_cudnn=self.use_cudnn,
                name='black_naive_long_effect_{}'.format(direction.name)
            )(board)

            long_effects[direction] = long_effect

            effect_list.append(tf.logical_or(short_effect, long_effect))

        eight_effects = CombineLayer()(effect_list)

        effect_ke_list = []
        for direction in (Direction.RIGHT_UP_UP, Direction.LEFT_UP_UP):
            effect_ke = BlackNaiveShortEffectLayer(
                direction=direction, data_format=self.data_format,
                use_cudnn=self.use_cudnn,
                name='black_naive_short_effect_{}'.format(direction)
            )(board)
            effect_ke_list.append(effect_ke)
        effect_ke = tf.logical_or(*effect_ke_list)

        effect = tf.logical_or(eight_effects, effect_ke)

        return effect, long_effects
