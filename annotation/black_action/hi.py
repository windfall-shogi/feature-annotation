#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from .major import BlackMajorMove
from ..boolean_board.empty import select_empty_board
from ..piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/2/24'


class BlackHiDropLayer(snt.AbstractModule):
    def __init__(self, name='black_hi_drop'):
        super().__init__(name=name)

    def _build(self, board, black_hand, available_square):
        empty_square = select_empty_board(board=board)

        available = tf.logical_and(
            # 空いているマス
            empty_square,
            # 持ち駒があるかどうか
            tf.reshape(
                tf.greater_equal(black_hand[:, Piece.BLACK_HI], 1),
                [-1, 1, 1, 1]
            )
        )
        # 王手の時に有効かどうか
        available = tf.logical_and(available, available_square)

        return available


class BlackHiMoveLayer(snt.AbstractModule):
    def __init__(self, data_format, name='black_hi_move'):
        super().__init__(name=name)
        self.data_format = data_format

    def _build(self, board, hi_effect):
        return BlackMajorMove(data_format=self.data_format)(board, hi_effect)
