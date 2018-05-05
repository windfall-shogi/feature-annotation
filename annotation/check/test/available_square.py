#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product, chain, repeat
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.black_effect.pseudo_ou_effect import BlackPseudoOuEffect
from annotation.direction import (Direction, get_eight_directions,
                                  get_opposite_direction, get_cross_directions,
                                  get_diagonal_directions)
from annotation.piece import Piece
from ..white_all_check import WhiteAllCheckLayer
from ..white_short_check import WhiteShortCheckLayer
from ..black_available_square import CheckAvailableSquareLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/05'


class TestAvailableSquare(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_short_check(self):
        """
        短い利きで王手されている場合に王以外の駒が動ける場所の判定をテスト
        :return:
        """
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        ph = tf.placeholder(tf.int32, shape=shape)
        pseudo_effect = BlackPseudoOuEffect(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph)
        all_check, _ = WhiteAllCheckLayer()(ph, pseudo_effect)
        available_square = CheckAvailableSquareLayer()(pseudo_effect,
                                                       all_check)
        available_square = tf.squeeze(available_square)
        short_check = WhiteShortCheckLayer()(ph, pseudo_effect)

        with self.test_session() as sess:
            for direction in chain(get_eight_directions(),
                                   [Direction.RIGHT_UP_UP,
                                    Direction.LEFT_UP_UP]):
                for i, j, piece in product(range(9), range(9),
                                           range(Piece.WHITE_FU, Piece.EMPTY)):
                    piece = Piece(piece)

                    if piece == Piece.WHITE_OU:
                        # OUで王手はない
                        continue
                    # 長い利きはここではテストしない
                    if piece == Piece.WHITE_KY and direction == Direction.UP:
                        continue
                    elif (piece in (Piece.WHITE_KA, Piece.WHITE_UM) and
                            direction in get_diagonal_directions()):
                        continue
                    elif (direction in get_cross_directions() and
                            piece in (Piece.WHITE_HI, Piece.WHITE_RY)):
                        continue

                    x, y = self._get_position(direction=direction, i=i, j=j)
                    if x not in range(9) or y not in range(9):
                        continue

                    board[:] = Piece.EMPTY
                    if self.data_format == 'NCHW':
                        board[0, 0, i, j] = Piece.BLACK_OU
                        board[0, 0, x, y] = piece
                    else:
                        board[0, i, j, 0] = Piece.BLACK_OU
                        board[0, x, y, 0] = piece

                    with self.subTest(direction=direction, i=i, j=j,
                                      piece=piece):
                        square, check = sess.run(
                            [available_square, short_check[direction]],
                            feed_dict={ph: board}
                        )

                        self.assertTupleEqual(square.shape, (9, 9))

                        if check:
                            self.assertTrue(square[x, y])
                            square[x, y] = False
                            self.assertFalse(np.any(square))
                        else:
                            self.assertTrue(np.all(square))

    def test_long_check(self):
        """
        長い利きで王手されている場合に王以外の駒が動ける場所の判定のテスト
        :return:
        """
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        ph = tf.placeholder(tf.int32, shape=shape)
        pseudo_effect = BlackPseudoOuEffect(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph)
        all_check, long_check = WhiteAllCheckLayer()(ph, pseudo_effect)
        available_square = CheckAvailableSquareLayer()(pseudo_effect,
                                                       all_check)
        available_square = tf.squeeze(available_square)

        long_piece_list = [Piece.WHITE_KY, Piece.WHITE_KA, Piece.WHITE_HI,
                           Piece.WHITE_UM, Piece.WHITE_RY]

        with self.test_session() as sess:
            for tmp in product(get_eight_directions(), range(9), range(9),
                               range(1, 9), long_piece_list):
                direction, i, j, k, piece = tmp

                # 短い利きは含めない
                if (piece == Piece.WHITE_UM and
                        direction in get_cross_directions()):
                    continue
                elif (piece == Piece.WHITE_RY and
                        direction == get_diagonal_directions()):
                    continue

                x, y = self._get_position(direction=direction, i=i, j=j, k=k)
                if x not in range(9) or y not in range(9):
                    continue

                board[:] = Piece.EMPTY
                if self.data_format == 'NCHW':
                    board[0, 0, i, j] = Piece.BLACK_OU
                    board[0, 0, x, y] = piece
                else:
                    board[0, i, j, 0] = Piece.BLACK_OU
                    board[0, x, y, 0] = piece

                with self.subTest(direction=direction, i=i, j=j,
                                  piece=piece):
                    square, check = sess.run(
                        [available_square, long_check[direction]],
                        feed_dict={ph: board}
                    )

                    self.assertTupleEqual(square.shape, (9, 9))

                    if check:
                        for u, v in zip(*self._get_range(i=i, j=j, x=x, y=y)):
                            self.assertTrue(square[u, v])
                            square[u, v] = False
                        self.assertFalse(np.any(square))
                    else:
                        self.assertTrue(np.all(square))

    @staticmethod
    def _get_position(direction, i, j, k=1):
        table = np.array([
            [-1, -1],  # right up
            [-1, 0],  # right
            [-1, 1],  # right down
            [0, -1],  # up
            [0, 1],  # down
            [1, -1],  # left up
            [1, 0],  # left
            [1, 1],  # left down
            [-1, -2],  # right up up
            [1, -2],  # left up up
        ])

        return np.array([i, j]) + table[direction] * k

    @staticmethod
    def _get_range(i, j, x, y):
        def _get(m, n):
            if m == n:
                return repeat(m)
            elif m < n:
                return range(m, n)
            else:
                return range(n, m)

        return _get(i, x), _get(j, y)
