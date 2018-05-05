#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from .major import BlackMajorMove
from ..boolean_board.empty import select_empty_board
from ..piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/2/24'


class BlackKaDropLayer(snt.AbstractModule):
    def __init__(self, name='black_ka_drop'):
        super().__init__(name=name)

    def _build(self, board, black_hand, available_square):
        empty_square = select_empty_board(board=board)

        available = tf.logical_and(
            # 空いているマス
            empty_square,
            # 持ち駒があるかどうか
            tf.reshape(
                tf.greater_equal(black_hand[:, Piece.BLACK_KA], 1),
                [-1, 1, 1, 1]
            )
        )
        # 王手の時に有効かどうか
        available = tf.logical_and(available, available_square)

        return available


class BlackKaMoveLayer(snt.AbstractModule):
    def __init__(self, data_format, name='black_ka_move'):
        super().__init__(name=name)
        self.data_format = data_format

    def _build(self, board, ka_effect):
        return BlackMajorMove(data_format=self.data_format)(board, ka_effect)
