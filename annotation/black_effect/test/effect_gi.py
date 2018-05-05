#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.direction import Direction, PinDirection
from annotation.piece import Piece
from ..gi import BlackGiEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/10'


class TestGiEffect(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_gi_effect(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        # 適当なマスクを設定
        available_board = np.zeros(81, dtype=np.bool)
        available_board[::2] = True
        available_board = np.reshape(available_board, shape)

        ph = tf.placeholder(tf.int32, shape=shape)
        gi_effect = BlackGiEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph, available_board)
        # 余計な方向がないことを確認する
        self.assertEqual(5, len(gi_effect))
        # 利用しやすいように次元を下げる
        gi_effect = {key: tf.squeeze(value)
                     for key, value in gi_effect.items()}

        # Layerに渡したので、変更しても大丈夫
        # アクセスしやすいように次元を下げる
        available_board = np.squeeze(available_board)

        with self.test_session() as sess:
            for i, j in product(range(9), repeat=2):
                # (i, j)に駒を置く

                board[:] = Piece.EMPTY
                if self.data_format == 'NCHW':
                    board[0, 0, i, j] = Piece.BLACK_GI
                else:
                    board[0, i, j, 0] = Piece.BLACK_GI

                effect = sess.run(gi_effect, feed_dict={ph: board})
                for key, value in effect.items():
                    if key == Direction.RIGHT_UP:
                        x = i - 1
                        y = j - 1
                    elif key == Direction.RIGHT_DOWN:
                        x = i - 1
                        y = j + 1
                    elif key == Direction.UP:
                        x = i
                        y = j - 1
                    elif key == Direction.LEFT_UP:
                        x = i + 1
                        y = j - 1
                    elif key == Direction.LEFT_DOWN:
                        x = i + 1
                        y = j + 1
                    else:
                        raise ValueError(key)

                    with self.subTest(i=i, j=j, direction=key):
                        if (x in range(9) and y in range(9) and
                                available_board[x, y]):
                            self.assertTrue(value[x, y])
                            value[x, y] = False
                            self.assertFalse(np.all(value))
                        else:
                            self.assertFalse(np.all(value))

    def test_gi_pin(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        # ここではすべて有効に設定
        available_board = np.ones(shape, dtype=np.bool)

        ph = tf.placeholder(tf.int32, shape=shape)
        gi_effect = BlackGiEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph, available_board)
        # 余計な方向がないことを確認する
        self.assertEqual(5, len(gi_effect))
        # 利用しやすいように次元を下げる
        gi_effect = {key: tf.squeeze(value)
                     for key, value in gi_effect.items()}

        with self.test_session() as sess:
            for i, j in product(range(9), repeat=2):
                # (i, j)に駒を置く

                board[:] = Piece.EMPTY
                for pin_direction in PinDirection:
                    if pin_direction == PinDirection.SIZE:
                        continue

                    offset = Piece.SIZE + pin_direction * 14
                    if self.data_format == 'NCHW':
                        board[0, 0, i, j] = Piece.BLACK_GI + offset
                    else:
                        board[0, i, j, 0] = Piece.BLACK_GI + offset

                    effect = sess.run(gi_effect, feed_dict={ph: board})
                    for key, value in effect.items():
                        if key == Direction.RIGHT_UP:
                            x = i - 1
                            y = j - 1
                        elif key == Direction.RIGHT_DOWN:
                            x = i - 1
                            y = j + 1
                        elif key == Direction.UP:
                            x = i
                            y = j - 1
                        elif key == Direction.LEFT_UP:
                            x = i + 1
                            y = j - 1
                        elif key == Direction.LEFT_DOWN:
                            x = i + 1
                            y = j + 1
                        else:
                            raise ValueError(key)

                        with self.subTest(i=i, j=j, direction=key):
                            if (x in range(9) and y in range(9) and
                                    pin_direction == PinDirection[key.name]):
                                self.assertTrue(value[x, y])
                                value[x, y] = False
                                self.assertFalse(np.all(value))
                            else:
                                self.assertFalse(np.all(value))
