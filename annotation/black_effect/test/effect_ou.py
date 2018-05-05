#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.direction import (Direction, get_eight_directions,
                                  get_opposite_direction)
from annotation.piece import Piece
from ..ou import BlackOuEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/10'


class TestOuEffect(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_ou_effect_without_long_check(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        # 適当なマスクを設定
        white_effect_mask = np.zeros(81, dtype=np.bool)
        white_effect_mask[::2] = True
        white_effect_mask = np.reshape(white_effect_mask, shape)

        white_long_check = {direction: False
                            for direction in get_eight_directions()}

        ph = tf.placeholder(tf.int32, shape=shape)
        ou_effect = BlackOuEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph, white_effect_mask, white_long_check)
        # 余計な方向がないことを確認する
        self.assertEqual(8, len(ou_effect))
        # 利用しやすいように次元を下げる
        ou_effect = {key: tf.squeeze(value)
                     for key, value in ou_effect.items()}

        # Layerに渡したので、変更しても大丈夫
        # アクセスしやすいように次元を下げる
        white_effect_mask = np.squeeze(white_effect_mask)

        with self.test_session() as sess:
            for i, j in product(range(9), repeat=2):
                # (i, j)に駒を置く

                board[:] = Piece.EMPTY
                if self.data_format == 'NCHW':
                    board[0, 0, i, j] = Piece.BLACK_OU
                else:
                    board[0, i, j, 0] = Piece.BLACK_OU

                effect = sess.run(ou_effect, feed_dict={ph: board})
                for key, value in effect.items():
                    x, y = self.get_square(i=i, j=j, direction=key)

                    with self.subTest(i=i, j=j, direction=key):
                        if (x in range(9) and y in range(9) and
                                not white_effect_mask[x, y]):
                            self.assertTrue(value[x, y])
                            value[x, y] = False
                            self.assertFalse(np.all(value))
                        else:
                            self.assertFalse(np.all(value))

    def test_ou_effect_with_long_check(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        # 利きはない状態に設定
        white_effect_mask = np.zeros_like(board, dtype=np.bool)

        white_long_check = {direction: tf.placeholder(tf.bool, shape=[])
                            for direction in get_eight_directions()}

        ph = tf.placeholder(tf.int32, shape=shape)
        ou_effect = BlackOuEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph, white_effect_mask, white_long_check)
        # 余計な方向がないことを確認する
        self.assertEqual(8, len(ou_effect))
        # 利用しやすいように次元を下げる
        ou_effect = {key: tf.squeeze(value)
                     for key, value in ou_effect.items()}

        feed_dict = {value: False for value in white_long_check.values()}
        feed_dict[ph] = board

        with self.test_session() as sess:
            for check_direction in get_eight_directions():
                feed_dict[white_long_check[check_direction]] = True
                # check_directionの方向に駒があるので、逃げるときに問題になるのは反対方向
                opposite_direction = get_opposite_direction(
                    direction=check_direction
                )

                for i, j in product(range(9), repeat=2):
                    # (i, j)に駒を置く

                    board[:] = Piece.EMPTY
                    if self.data_format == 'NCHW':
                        board[0, 0, i, j] = Piece.BLACK_OU
                    else:
                        board[0, i, j, 0] = Piece.BLACK_OU

                    effect = sess.run(ou_effect, feed_dict=feed_dict)
                    for key, value in effect.items():
                        x, y = self.get_square(i=i, j=j, direction=key)

                        with self.subTest(i=i, j=j, direction=key,
                                          check_direction=check_direction):
                            if (x in range(9) and y in range(9) and
                                    key != opposite_direction):
                                self.assertTrue(value[x, y])
                                value[x, y] = False
                                self.assertFalse(np.all(value))
                            else:
                                self.assertFalse(np.all(value))

                feed_dict[white_long_check[check_direction]] = False

    @staticmethod
    def get_square(i, j, direction):
        if direction == Direction.RIGHT_UP:
            x = i - 1
            y = j - 1
        elif direction == Direction.RIGHT:
            x = i - 1
            y = j
        elif direction == Direction.RIGHT_DOWN:
            x = i - 1
            y = j + 1
        elif direction == Direction.UP:
            x = i
            y = j - 1
        elif direction == Direction.DOWN:
            x = i
            y = j + 1
        elif direction == Direction.LEFT_UP:
            x = i + 1
            y = j - 1
        elif direction == Direction.LEFT:
            x = i + 1
            y = j
        elif direction == Direction.LEFT_DOWN:
            x = i + 1
            y = j + 1
        else:
            raise ValueError(direction)

        return x, y
