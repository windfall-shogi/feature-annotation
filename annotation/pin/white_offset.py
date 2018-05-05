#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from ..boolean_board import select_white_all_board
from ..direction import (get_eight_directions, get_opposite_direction,
                         PinDirection)
from ..piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/2/18'


class WhitePinLayer(snt.AbstractModule):
    def __init__(self, data_format, name='white_pin'):
        super().__init__(name=name)
        self.data_format = data_format

    def _build(self, board, white_pseudo_ou_effect, black_long_effect):
        """

        :param board:
        :param white_pseudo_ou_effect:
        :param black_long_effect:
        :return:
        """
        pin_value_list = [board]
        white_piece = select_white_all_board(board=board)
        for direction in get_eight_directions():
            # 反対同士からの利きが当たっている駒はピンされている
            flag = tf.logical_and(
                white_pseudo_ou_effect[direction],
                black_long_effect[get_opposite_direction(direction)]
            )
            # ピンされているならば、そのマスだけがTrue
            pinned = tf.logical_and(flag, white_piece)

            # ピンの方向に対応した値を計算
            offset = (Piece.SIZE - Piece.WHITE_FU +
                      14 * PinDirection[direction.name])
            pin_value_list.append(offset * tf.to_int32(pinned))

        # ピンされているマスに方向に応じた値を加える
        pinned_board = tf.add_n(pin_value_list)
        return pinned_board
