#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from .drop_mask import make_drop_mask1
from .promotion_mask import make_promotion_mask
from ..boolean_board.black import select_non_black_board
from ..boolean_board.empty import select_empty_board
from ..direction import Direction
from ..piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/2/23'


class BlackKyDropLayer(snt.AbstractModule):
    def __init__(self, data_format, name='black_ky_drop'):
        super().__init__(name=name)
        self.data_format = data_format

    def _build(self, board, black_hand, available_square):
        ky_available_area = make_drop_mask1(data_format=self.data_format)
        empty_square = select_empty_board(board=board)

        available = tf.logical_and(
            # 置けるマスを判定
            tf.logical_and(ky_available_area, empty_square),
            # 持ち駒があるかどうか
            tf.reshape(
                tf.greater_equal(black_hand[:, Piece.BLACK_KY], 1),
                [-1, 1, 1, 1]
            )
        )
        # 王手の時に有効かどうか
        available = tf.logical_and(available, available_square)

        return available


class BlackKyMoveLayer(snt.AbstractModule):
    def __init__(self, data_format, name='black_ky_move'):
        super().__init__(name=name)
        self.data_format = data_format

    def _build(self, board, ky_effect):
        non_black_mask = select_non_black_board(board=board)
        movable_effect = [tf.logical_and(effect, non_black_mask)
                          for effect in ky_effect[Direction.UP]]

        available_mask = make_drop_mask1(data_format=self.data_format)
        non_promoting_effect = {
            Direction.UP: [tf.logical_and(effect, available_mask)
                           for effect in movable_effect]
        }

        # 上方向はどのステップサイズでも同じ
        promotion_mask = make_promotion_mask(
            direction=Direction.UP, data_format=self.data_format, step_size=1
        )
        promoting_effect = {
            Direction.UP: [tf.logical_and(effect, promotion_mask)
                           for effect in movable_effect]
        }

        return non_promoting_effect, promoting_effect
