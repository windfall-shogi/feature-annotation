#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from ..boolean_board.black import select_non_black_board
from ..boolean_board.empty import select_empty_board
from ..piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/2/24'


class BlackKiDropLayer(snt.AbstractModule):
    def __init__(self, name='black_ki_drop'):
        super().__init__(name=name)

    def _build(self, board, black_hand, available_square):
        empty_square = select_empty_board(board=board)

        available = tf.logical_and(
            # 空いているマス
            empty_square,
            # 持ち駒があるかどうか
            tf.reshape(
                tf.greater_equal(black_hand[:, Piece.BLACK_KI], 1),
                [-1, 1, 1, 1]
            )
        )
        # 王手の時に有効かどうか
        available = tf.logical_and(available, available_square)

        return available


class BlackKiMoveLayer(snt.AbstractModule):
    def __init__(self, name='black_ki_move'):
        super().__init__(name=name)

    def _build(self, board, ki_effect):
        non_black_mask = select_non_black_board(board=board)
        non_promoting_effect = {
            direction: tf.logical_and(non_black_mask, effect)
            for direction, effect in ki_effect.items()
        }

        return non_promoting_effect
