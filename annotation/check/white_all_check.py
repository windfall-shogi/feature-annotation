#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from ..direction import Direction, get_eight_directions
from .white_long_check import WhiteLongCheckLayer
from .white_short_check import WhiteShortCheckLayer

__author__ = 'Yasuhiro'
__date__ = '2018/2/17'


class WhiteAllCheckLayer(snt.AbstractModule):
    def __init__(self, name='white_all_check'):
        super().__init__(name=name)

    def _build(self, board, pseudo_ou_effects):
        """
        長い利きでの王手と短い利きでの王手を組み合わせて、全ての王手を計算する

        :param board:
        :param pseudo_ou_effects: 短い利きと長い利きで分かれている namedtuple
        :return:
        """
        short_check = WhiteShortCheckLayer()(board, pseudo_ou_effects.short)
        long_check = WhiteLongCheckLayer()(board, pseudo_ou_effects.long)

        outputs = {
            direction: tf.logical_or(short_check[direction],
                                     long_check[direction])
            for direction in get_eight_directions()
        }
        outputs.update({
            direction: short_check[direction]
            for direction in (Direction.RIGHT_UP_UP, Direction.LEFT_UP_UP)
        })

        return outputs, long_check
