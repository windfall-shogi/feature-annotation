#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.direction import Direction, PinDirection
from annotation.piece import Piece
from ..fu import BlackFuEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/07'


class TestFuEffect(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_fu_effect(self):
        """
        王手がかかっている場合に移動できる場所に制限がかかるので、
        それに対応しているかを確認する
        :return:
        """
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        # 適当なマスクを設定
        available_board = np.zeros(81, dtype=np.bool)
        available_board[::2] = True
        available_board = np.reshape(available_board, shape)

        ph = tf.placeholder(tf.int32, shape=shape)
        fu_effect = BlackFuEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph, available_board)
        # 余計な方向がないことを確認する
        self.assertEqual(1, len(fu_effect))
        fu_effect = tf.squeeze(fu_effect[Direction.UP])

        # Layerに渡したので、変更しても大丈夫
        # アクセスしやすいように次元を下げる
        available_board = np.squeeze(available_board)

        with self.test_session() as sess:
            for i, j in product(range(9), repeat=2):
                # (i, j)に駒を置く、(i, j - 1)が利きのある位置
                if j == 0:
                    continue

                board[:] = Piece.EMPTY
                if self.data_format == 'NCHW':
                    board[0, 0, i, j] = Piece.BLACK_FU
                else:
                    board[0, i, j, 0] = Piece.BLACK_FU

                effect = sess.run(fu_effect, feed_dict={ph: board})
                with self.subTest(i=i, j=j):
                    if available_board[i, j - 1]:
                        self.assertTrue(effect[i, j - 1])
                        effect[i, j - 1] = False
                        self.assertFalse(np.all(effect))
                    else:
                        self.assertFalse(np.all(effect))

    def test_fu_pin(self):
        """
        ピンされている場合に制限がかかるので、それに対応しているかを確認する
        :return:
        """
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        # ここではすべて有効にする
        available_board = np.ones(shape, dtype=np.bool)

        ph = tf.placeholder(tf.int32, shape=shape)
        fu_effect = BlackFuEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph, available_board)
        # 余計な方向がないことを確認する
        self.assertEqual(1, len(fu_effect))
        fu_effect = tf.squeeze(fu_effect[Direction.UP])

        with self.test_session() as sess:
            for i, j in product(range(9), repeat=2):
                # (i, j)に駒を置く、(i, j - 1)が利きのある位置
                if j == 0:
                    continue

                board[:] = Piece.EMPTY
                for pin_direction in PinDirection:
                    if pin_direction == PinDirection.SIZE:
                        continue

                    offset = Piece.SIZE + pin_direction * 14
                    if self.data_format == 'NCHW':
                        board[0, 0, i, j] = Piece.BLACK_FU + offset
                    else:
                        board[0, i, j, 0] = Piece.BLACK_FU + offset

                    effect = sess.run(fu_effect, feed_dict={ph: board})
                    with self.subTest(i=i, j=j, pin_direction=pin_direction):
                        if pin_direction == PinDirection.VERTICAL:
                            self.assertTrue(effect[i, j - 1])
                            effect[i, j - 1] = False
                            self.assertFalse(np.all(effect))
                        else:
                            self.assertFalse(np.all(effect))
