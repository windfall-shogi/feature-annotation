#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.direction import (Direction, get_diagonal_directions,
                                  PinDirection)
from annotation.piece import Piece
from ..um import BlackUmEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/10'


class TestUmEffect(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_um_effect(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        # 適当なマスクを設定
        available_board = np.zeros(81, dtype=np.bool)
        available_board[::2] = True
        available_board = np.reshape(available_board, shape)

        ph = tf.placeholder(tf.int32, shape=shape)
        um_effect = BlackUmEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph, available_board)
        # 余計な方向がないことを確認する
        self.assertEqual(8, len(um_effect))
        # 利用しやすいように次元を下げる
        um_effect = {direction: tf.squeeze(effect)
                     for direction, effect in um_effect.items()}
        # 効きが長い方向のみ8種類の距離を一つにまとめる
        diagonal_directions = get_diagonal_directions()
        um_effect = {
            direction: tf.reduce_any(um_effect[direction], axis=0)
            if direction in diagonal_directions else effect
            for direction, effect in um_effect.items()
        }

        # Layerに渡したので、変更しても大丈夫
        # アクセスしやすいように次元を下げる
        available_board = np.squeeze(available_board)

        with self.test_session() as sess:
            for i, j in product(range(9), repeat=2):
                # (i, j)に駒を置く

                board[:] = Piece.EMPTY
                if self.data_format == 'NCHW':
                    board[0, 0, i, j] = Piece.BLACK_UM
                else:
                    board[0, i, j, 0] = Piece.BLACK_UM

                effect = sess.run(um_effect, feed_dict={ph: board})
                for direction, e in effect.items():
                    with self.subTest(i=i, j=j, direction=direction):
                        s, t = self._get_range(i=i, j=j, direction=direction)

                        for x, y in zip(s, t):
                            if x not in range(9) or y not in range(9):
                                # 盤の外に到達した
                                break

                            if available_board[x, y]:
                                self.assertTrue(e[x, y])
                                # 利きをリセット
                                e[x, y] = False
                        # 最後にまとめて一つも利きがないことを確認する
                        self.assertFalse(np.all(e))

    def test_um_pin(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        # ここではすべて有効に設定
        available_board = np.ones(shape, dtype=np.bool)

        ph = tf.placeholder(tf.int32, shape=shape)
        um_effect = BlackUmEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph, available_board)
        # 余計な方向がないことを確認する
        self.assertEqual(8, len(um_effect))
        # 利用しやすいように次元を下げる
        um_effect = {direction: tf.squeeze(effect)
                     for direction, effect in um_effect.items()}
        # 効きが長い方向のみ8種類の距離を一つにまとめる
        diagonal_directions = get_diagonal_directions()
        um_effect = {
            direction: tf.reduce_any(um_effect[direction], axis=0)
            if direction in diagonal_directions else effect
            for direction, effect in um_effect.items()
        }

        with self.test_session() as sess:
            for i, j in product(range(9), repeat=2):
                # (i, j)に駒を置く

                board[:] = Piece.EMPTY
                for pin_direction in PinDirection:
                    if pin_direction == PinDirection.SIZE:
                        continue

                    if self.data_format == 'NCHW':
                        board[0, 0, i, j] = Piece.BLACK_UM
                    else:
                        board[0, i, j, 0] = Piece.BLACK_UM

                    effect = sess.run(um_effect, feed_dict={ph: board})
                    for direction, e in effect.items():
                        with self.subTest(i=i, j=j, direction=direction):
                            s, t = self._get_range(i=i, j=j,
                                                   direction=direction)

                            for x, y in zip(s, t):
                                if x not in range(9) or y not in range(9):
                                    # 盤の外に到達した
                                    break

                                if (pin_direction ==
                                        PinDirection[direction.name]):
                                    self.assertTrue(e[x, y])
                                    # 利きをリセット
                                    e[x, y] = False
                            # 最後にまとめて一つも利きがないことを確認する
                            self.assertFalse(np.all(e))

    @staticmethod
    def _get_range(i, j, direction):
        # zipで短い方に合わせるので、駒に近い方から座標を生成する
        if direction == Direction.RIGHT_UP:
            s = range(i - 1, -1, -1)
            t = range(j - 1, -1, -1)
        elif direction == Direction.RIGHT:
            s = range(i - 1, i - 2, -1)
            t = [j]
        elif direction == Direction.RIGHT_DOWN:
            s = range(i - 1, -1, -1)
            t = range(j + 1, 9)
        elif direction == Direction.UP:
            s = [i]
            t = range(j - 1, j - 2, -1)
        elif direction == Direction.DOWN:
            s = [i]
            t = range(j + 1, j + 2)
        elif direction == Direction.LEFT_UP:
            s = range(i + 1, 9)
            t = range(j - 1, -1, -1)
        elif direction == Direction.LEFT:
            s = range(i + 1, i + 2)
            t = [j]
        elif direction == Direction.LEFT_DOWN:
            s = range(i + 1, 9)
            t = range(j + 1, 9)
        else:
            raise ValueError(direction)

        return s, t
