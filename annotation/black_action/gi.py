#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from .promotion_mask import make_promotion_mask
from ..boolean_board.black import select_non_black_board
from ..boolean_board.empty import select_empty_board
from ..piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/2/23'


class BlackGiDropLayer(snt.AbstractModule):
    def __init__(self, name='black_gi_drop'):
        super().__init__(name=name)

    def _build(self, board, black_hand, available_square):
        empty_square = select_empty_board(board=board)

        available = tf.logical_and(
            # 空いているマス
            empty_square,
            # 持ち駒があるかどうか
            tf.reshape(
                tf.greater_equal(black_hand[:, Piece.BLACK_GI], 1),
                [-1, 1, 1, 1]
            )
        )
        # 王手の時に有効かどうか
        available = tf.logical_and(available, available_square)

        return available


class BlackGiMoveLayer(snt.AbstractModule):
    def __init__(self, data_format, name='black_ki_move'):
        super().__init__(name=name)
        self.data_format = data_format

    def _build(self, board, gi_effect):
        non_black_mask = select_non_black_board(board=board)
        non_promoting_effect = {
            direction: tf.logical_and(non_black_mask, effect)
            for direction, effect in gi_effect.items()
        }

        promoting_effect = {
            direction: tf.logical_and(
                make_promotion_mask(
                    direction=direction, data_format=self.data_format,
                    step_size=1
                ),
                effect
            )
            for direction, effect in non_promoting_effect.items()
        }

        return non_promoting_effect, promoting_effect
