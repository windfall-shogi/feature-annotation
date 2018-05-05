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
                                  get_opposite_direction, get_cross_directions,
                                  get_diagonal_directions)
from annotation.piece import Piece
from ..white_long_check import WhiteLongCheckLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/04'


class TestCheckLong(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_check(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        ph = tf.placeholder(tf.int32, shape=shape)
        pseudo_effect = BlackPseudoOuEffect(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph)
        long_check = WhiteLongCheckLayer()(ph, pseudo_effect)

        # 利きの長い駒のみに限定する
        # UM,RYの長い利きのみで、短い利きは対象に含めない
        white_pieces = [Piece.WHITE_KY, Piece.WHITE_KA, Piece.WHITE_HI,
                        Piece.WHITE_UM, Piece.WHITE_RY]

        for direction in get_eight_directions():
            with self.test_session() as sess:
                for i, j, k, piece in product(range(9), range(9), range(1, 9),
                                              white_pieces):
                    x, y = self._get_position(
                        direction=direction, i=i, j=j, k=k
                    )
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

                    with self.subTest(direction=direction, i=i, j=j, k=k,
                                      piece=piece):
                        check = sess.run(long_check, feed_dict={ph: board})

                        for key, value in check.items():
                            # ブロードキャストが正しく行われるために4次元が必要
                            self.assertTupleEqual((1, 1, 1, 1), value.shape)

                            if key == direction:
                                d = get_opposite_direction(direction=direction)
                                if piece == Piece.WHITE_KY:
                                    self.assertEqual(value,
                                                     d == Direction.DOWN)
                                elif piece in (Piece.WHITE_KA, Piece.WHITE_UM):
                                    self.assertEqual(
                                        value, d in get_diagonal_directions()
                                    )
                                elif piece in (Piece.WHITE_HI, Piece.WHITE_RY):
                                    self.assertEqual(
                                        value, d in get_cross_directions()
                                    )
                                else:
                                    # ここには到達しないはず
                                    raise ValueError()
                            else:
                                self.assertFalse(value)

    @staticmethod
    def _get_position(direction, i, j, k):
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

        return np.array([i, j]) + table[direction] * k
