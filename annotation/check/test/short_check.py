#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from itertools import product, chain
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.black_effect.pseudo_ou_effect import BlackPseudoOuEffect
from annotation.direction import (Direction, get_eight_directions,
                                  get_opposite_direction)
from annotation.piece import Piece
from annotation.short_board.white_table import make_base_table
from ..white_short_check import WhiteShortCheckLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/04'


class TestCheckShort(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_check(self):
        """
        短い利きでの王手があるかの判定のテスト

        :return:
        """
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        ph = tf.placeholder(tf.int32, shape=shape)
        pseudo_effect = BlackPseudoOuEffect(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph)
        short_check = WhiteShortCheckLayer()(ph, pseudo_effect)

        # 利きの短い駒のみに限定する
        # UM,RYの短い利きのみで、長い利きは対象に含めない
        white_pieces = [
            Piece.WHITE_FU, Piece.WHITE_KE, Piece.WHITE_GI, Piece.WHITE_KI,
            Piece.WHITE_TO, Piece.WHITE_NY, Piece.WHITE_NK, Piece.WHITE_NG,
            Piece.WHITE_UM, Piece.WHITE_RY
        ]
        direction_table = make_base_table()

        for direction in chain(get_eight_directions(),
                               [Direction.RIGHT_UP_UP, Direction.LEFT_UP_UP]):
            with self.test_session() as sess:
                for i, j, piece in product(range(9), range(9), white_pieces):
                    x, y = self._get_position(direction=direction, i=i, j=j)
                    if x not in range(9) or y not in range(9):
                        continue

                    # OUから見てdirectionの方向に相手の駒を配置
                    board[:] = Piece.EMPTY
                    if self.data_format == 'NCHW':
                        board[0, 0, i, j] = Piece.BLACK_OU
                        board[0, 0, x, y] = piece
                    else:
                        board[0, i, j, 0] = Piece.BLACK_OU
                        board[0, x, y, 0] = piece

                    with self.subTest(direction=direction, i=i, j=j,
                                      piece=piece):
                        check = sess.run(short_check, feed_dict={ph: board})

                        for key, value in check.items():
                            # ブロードキャストが正しく行われるために4次元が必要
                            self.assertTupleEqual((1, 1, 1, 1), value.shape)

                            if key == direction:
                                d = get_opposite_direction(direction=direction)
                                index = piece - Piece.WHITE_FU
                                flag = direction_table[d][index]
                                self.assertEqual(value, bool(flag))
                            else:
                                self.assertFalse(value)
            pass

    @staticmethod
    def _get_position(direction, i, j):
        table = np.array([
            [-1, -1],  # right up
            [-1, 0],   # right
            [-1, 1],   # right down
            [0, -1],   # up
            [0, 1],    # down
            [1, -1],   # left up
            [1, 0],    # left
            [1, 1],    # left down
            [-1, -2],  # right up up
            [1, -2],   # left up up
        ])

        return np.array([i, j]) + table[direction]
