#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from .drop_mask import make_drop_mask1
from .promotion_mask import make_promotion_mask
from ..boolean_board.black import select_black_fu_board, select_non_black_board
from ..boolean_board.empty import select_empty_board
from ..direction import Direction
from ..piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/2/22'


class BlackFuFileLayer(snt.AbstractModule):
    def __init__(self, data_format, name='black_fu_file'):
        super().__init__(name=name)
        self.data_format = data_format

    def _build(self, board):
        fu_board = select_black_fu_board(board=board)

        axis = -1 if self.data_format == 'NCHW' else -2
        flag = tf.reduce_any(fu_board, axis=axis, keep_dims=True)
        flag = tf.logical_not(flag)
        repeat_count = [1, 1, 1, 1]
        repeat_count[axis] = 9
        available_map = tf.tile(flag, repeat_count)

        return available_map


class BlackFuDropLayer(snt.AbstractModule):
    def __init__(self, data_format, name='black_fu_drop'):
        super().__init__(name=name)
        self.data_format = data_format

    def _build(self, board, black_hand, available_square):
        fu_available_file = BlackFuFileLayer(
            data_format=self.data_format
        )(board)
        fu_available_area = make_drop_mask1(data_format=self.data_format)

        empty_square = select_empty_board(board=board)

        available = tf.logical_and(
            # FUを置ける筋、2~9段
            tf.logical_and(fu_available_file, fu_available_area),
            tf.logical_and(
                # 空いているマス
                empty_square,
                # 持ち駒があるかどうか
                tf.reshape(
                    tf.greater_equal(black_hand[:, Piece.BLACK_FU], 1),
                    [-1, 1, 1, 1]
                )
            )
        )
        # 王手の時に有効かどうか
        available = tf.logical_and(available, available_square)

        return available


class BlackFuMoveLayer(snt.AbstractModule):
    def __init__(self, data_format, name='black_fu_move'):
        super().__init__(name=name)
        self.data_format = data_format

    def _build(self, board, fu_effect):
        non_black_mask = select_non_black_board(board=board)
        movable_effect = tf.logical_and(fu_effect[Direction.UP],
                                        non_black_mask)

        available_mask = make_drop_mask1(data_format=self.data_format)
        non_promoting_effect = {
            Direction.UP: tf.logical_and(movable_effect, available_mask)
        }

        promotion_mask = make_promotion_mask(
            direction=Direction.UP, data_format=self.data_format, step_size=1
        )
        promoting_effect = {
            Direction.UP: tf.logical_and(movable_effect, promotion_mask)
        }

        return non_promoting_effect, promoting_effect
