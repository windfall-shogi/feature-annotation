#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import sonnet as snt

from ..boolean_board import white
from ..direction import Direction

__author__ = 'Yasuhiro'
__date__ = '2018/2/17'


class WhiteLongCheckLayer(snt.AbstractModule):
    def __init__(self, name='white_long_check'):
        super().__init__(name=name)

    def _build(self, board, pseudo_ou_effects):
        """
        利きの長い駒で王手しているかを判定する
        王手の場合は擬似的な王の利きの範囲に他の駒を動かせられれ王手を防げるので、
        pseudo_ou_effectsを利用して判定する

        王手がかかっている可能性があるのは手番側のみなのなので、逆のパターンはない

        :param board:
        :param pseudo_ou_effects:
        :return:
        """
        cross_pieces = white.select_white_hi_ry_board(board=board)
        diagonal_pieces = white.select_white_ka_um_board(board=board)
        ky_pieces = white.select_white_ky_board(board=board)

        outputs = {}
        for direction, pseudo_effect in pseudo_ou_effects.items():
            if direction == Direction.UP:
                pieces = tf.logical_or(cross_pieces, ky_pieces)
            elif direction in (Direction.RIGHT, Direction.DOWN,
                               Direction.LEFT):
                pieces = cross_pieces
            else:
                pieces = diagonal_pieces

            pseudo_effect = pseudo_ou_effects[direction]

            check = tf.logical_and(pseudo_effect, pieces)
            flag = tf.reduce_any(check, axis=[1, 2, 3], keep_dims=True)
            outputs[direction] = flag

        return outputs
