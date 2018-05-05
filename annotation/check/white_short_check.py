#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain

import tensorflow as tf
import sonnet as snt

from ..boolean_board import white
from ..direction import Direction, get_eight_directions, get_opposite_direction

__author__ = 'Yasuhiro'
__date__ = '2018/2/17'


class WhiteShortCheckLayer(snt.AbstractModule):
    def __init__(self, name='white_short_check'):
        super().__init__(name=name)

    def _build(self, board, pseudo_ou_effects):
        """
        利きの短い駒で王手しているかを判定する
        王手の場合は擬似的な王の利きの範囲に他の駒を動かせられれ王手を防げるので、
        pseudo_ou_effectsを利用して判定する

        王手がかかっている可能性があるのは手番側のみなのなので、逆のパターンはない

        :param board:
        :param pseudo_ou_effects:
        :return:
        """
        outputs = {}
        for direction, pseudo_effect in pseudo_ou_effects.items():
            # OUの通常の動きの8方向も含まれているが、まとめてpseudo_effectと呼ぶ
            opposite_direction = get_opposite_direction(direction=direction)
            piece = white.select_white_direction_board(
                board=board, direction=opposite_direction
            )

            pseudo_effect = pseudo_ou_effects[direction]

            check = tf.logical_and(pseudo_effect, piece)
            flag = tf.reduce_any(check, axis=[1, 2, 3], keep_dims=True)
            outputs[direction] = flag

        return outputs
