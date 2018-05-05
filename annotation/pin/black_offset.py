#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from ..boolean_board import select_black_all_board
from ..direction import (get_eight_directions, get_opposite_direction,
                         PinDirection)
from ..piece import Piece
__author__ = 'Yasuhiro'
__date__ = '2018/2/17'


class BlackPinLayer(snt.AbstractModule):
    def __init__(self, data_format, name='black_pin'):
        super().__init__(name=name)
        self.data_format = data_format

    def _build(self, board, black_pseudo_ou_effect, white_long_effect):
        """

        :param board:
        :param black_pseudo_ou_effect:
        :param white_long_effect:
        :return:
        """
        # もともとのboardの値を初期値としてセット
        pin_value_list = [board]

        black_piece = select_black_all_board(board=board)
        for direction in get_eight_directions():
            # 反対同士からの利きが当たっている駒はピンされている
            flag = tf.logical_and(
                black_pseudo_ou_effect[direction],
                white_long_effect[get_opposite_direction(direction)]
            )
            # 王手されている場合の経路の可能性もある
            # ピンならば、駒がある
            pinned = tf.logical_and(flag, black_piece)

            # ピンされているならば、そのマスだけがTrue
            # ピンの方向に対応した値を計算
            offset = Piece.SIZE + 14 * PinDirection[direction.name]
            pin_value_list.append(offset * tf.to_int32(pinned))

        # ピンされているマスに方向に応じた値を加える
        pinned_board = tf.add_n(pin_value_list)
        return pinned_board
