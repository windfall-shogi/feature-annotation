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
from ..ky import BlackKyEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/08'


class TestKyEffect(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_ky_effect(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        # 適当なマスクを設定
        available_board = np.zeros(81, dtype=np.bool)
        available_board[::2] = True
        available_board = np.reshape(available_board, shape)

        ph = tf.placeholder(tf.int32, shape=shape)
        ky_effect = BlackKyEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph, available_board)
        # 余計な方向がないことを確認する
        self.assertEqual(1, len(ky_effect))
        # 利用しやすいように次元を下げる
        ky_effect = tf.squeeze(ky_effect[Direction.UP])
        # 8種類の距離を一つにまとめる
        ky_effect = tf.reduce_any(ky_effect, axis=0)

        # Layerに渡したので、変更しても大丈夫
        # アクセスしやすいように次元を下げる
        available_board = np.squeeze(available_board)

        with self.test_session() as sess:
            for i, j in product(range(9), repeat=2):
                # (i, j)に駒を置く、(i, 0:j)が利きのある位置
                if j == 0:
                    continue

                board[:] = Piece.EMPTY
                if self.data_format == 'NCHW':
                    board[0, 0, i, j] = Piece.BLACK_KY
                else:
                    board[0, i, j, 0] = Piece.BLACK_KY

                effect = sess.run(ky_effect, feed_dict={ph: board})
                with self.subTest(i=i, j=j):
                    for k in range(j):
                        if available_board[i, k]:
                            self.assertTrue(effect[i, k])
                            # 利きをリセット
                            effect[i, k] = False
                    # 最後にまとめて一つも利きがないことを確認する
                    self.assertFalse(np.all(effect))

    def test_ky_pin(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        # ここではすべて有効に設定
        available_board = np.ones(shape, dtype=np.bool)

        ph = tf.placeholder(tf.int32, shape=shape)
        ky_effect = BlackKyEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph, available_board)
        # 余計な方向がないことを確認する
        self.assertEqual(1, len(ky_effect))
        # 利用しやすいように次元を下げる
        ky_effect = tf.squeeze(ky_effect[Direction.UP])
        # 8種類の距離を一つにまとめる
        ky_effect = tf.reduce_any(ky_effect, axis=0)

        with self.test_session() as sess:
            for i, j in product(range(9), repeat=2):
                # (i, j)に駒を置く、(i, 0:j)が利きのある位置
                if j == 0:
                    continue

                board[:] = Piece.EMPTY
                for pin_direction in PinDirection:
                    if pin_direction == PinDirection.SIZE:
                        continue

                    offset = Piece.SIZE + pin_direction * 14
                    if self.data_format == 'NCHW':
                        board[0, 0, i, j] = Piece.BLACK_KY + offset
                    else:
                        board[0, i, j, 0] = Piece.BLACK_KY + offset

                    effect = sess.run(ky_effect, feed_dict={ph: board})
                    with self.subTest(i=i, j=j, pin_direction=pin_direction):
                        for k in range(j):
                            if pin_direction == PinDirection.VERTICAL:
                                self.assertTrue(effect[i, k])
                                # 利きをリセット
                                effect[i, k] = False
                        # 最後にまとめて一つも利きがないことを確認する
                        self.assertFalse(np.all(effect))
