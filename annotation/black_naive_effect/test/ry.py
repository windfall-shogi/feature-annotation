#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from itertools import product, chain
from operator import add, sub

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.piece import Piece
from annotation.direction import (Direction, get_eight_directions,
                                  get_diagonal_directions, get_cross_directions)

from ..naive_long import BlackNaiveLongEffectLayer
from ..naive_short import BlackNaiveShortEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/18'


class TestWhiteLongEffectRy(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_effect1_short(self):
        """
        RYの短い利きをテスト

        :return:
        """
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        ph = tf.placeholder(tf.int32, shape=shape)
        for direction in chain(get_eight_directions(),
                               [Direction.RIGHT_UP_UP, Direction.LEFT_UP_UP]):
            white_effect = BlackNaiveShortEffectLayer(
                direction=direction, data_format=self.data_format,
                use_cudnn=self.use_cudnn
            )(ph)
            # チャネルの処理が面倒なので、次元を下げる
            white_effect = tf.squeeze(white_effect)

            with self.test_session() as sess:
                for i, j in product(range(9), repeat=2):
                    with self.subTest(direction=direction, i=i, j=j):
                        board[:] = Piece.EMPTY
                        if self.data_format == 'NCHW':
                            board[0, 0, i, j] = Piece.BLACK_RY
                        else:
                            board[0, i, j, 0] = Piece.BLACK_RY

                        effect = sess.run(white_effect, feed_dict={ph: board})

                        self.assertTupleEqual(effect.shape, (9, 9))

                        if direction not in get_diagonal_directions():
                            # 利きがあるマスはない
                            self.assertFalse(np.any(effect))
                            continue

                        if direction == Direction.RIGHT_UP:
                            u, v = i - 1, j - 1
                        elif direction == Direction.RIGHT_DOWN:
                            u, v = i - 1, j + 1
                        elif direction == Direction.LEFT_UP:
                            u, v = i + 1, j - 1
                        elif direction == Direction.LEFT_DOWN:
                            u, v = i + 1, j + 1
                        else:
                            # ここには到達しないはず
                            raise ValueError()
                        if u not in range(9) or v not in range(9):
                            # 盤の端に駒があるので、盤の中に利きはない
                            self.assertFalse(np.any(effect))
                            continue

                        self.assertTrue(effect[u, v])
                        # 利きがあるのは1か所だけのはず
                        effect[u, v] = False
                        self.assertFalse(np.any(effect))

    def test_effect1_long(self):
        """
        KAの利きがあるかを確認するテスト
        他の駒が利きを遮る場合は考えない

        :return:
        """
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        ph = tf.placeholder(tf.int32, shape=shape)
        for direction in chain(get_eight_directions(),
                               [Direction.RIGHT_UP_UP, Direction.LEFT_UP_UP]):
            if direction in (Direction.RIGHT_UP_UP, Direction.LEFT_UP_UP):
                # 桂馬の方向の長い利きはあり得ないのでエラー
                with self.assertRaises(ValueError):
                    BlackNaiveLongEffectLayer(
                        direction=direction, data_format=self.data_format,
                        use_cudnn=self.use_cudnn
                    )(ph)
                continue

            white_effect = BlackNaiveLongEffectLayer(
                direction=direction, data_format=self.data_format,
                use_cudnn=self.use_cudnn
            )(ph)
            # チャネルの処理が面倒なので、次元を下げる
            white_effect = tf.squeeze(white_effect)

            with self.test_session() as sess:
                for i, j in product(range(9), repeat=2):
                    with self.subTest(direction=direction, i=i, j=j):
                        board[:] = Piece.EMPTY
                        if self.data_format == 'NCHW':
                            board[0, 0, i, j] = Piece.BLACK_RY
                        else:
                            board[0, i, j, 0] = Piece.BLACK_RY

                        effect = sess.run(white_effect, feed_dict={ph: board})

                        self.assertTupleEqual(effect.shape, (9, 9))

                        if direction not in get_cross_directions():
                            # 利きがあるマスはない
                            self.assertFalse(np.any(effect))
                            continue

                        if direction == Direction.RIGHT:
                            edge = i == 0
                        elif direction == Direction.LEFT:
                            edge = i == 8
                        elif direction == Direction.UP:
                            edge = j == 0
                        elif direction == Direction.DOWN:
                            edge = j == 8
                        else:
                            raise ValueError(direction)

                        if edge:
                            # 盤の端に駒があるので、盤の中に利きはない
                            self.assertFalse(np.any(effect))
                            continue

                        if direction == Direction.RIGHT:
                            self.assertTrue(np.all(effect[:i, j]))
                            effect[:i, j] = False
                        elif direction == Direction.LEFT:
                            self.assertTrue(np.all(effect[i + 1:, j]))
                            effect[i + 1:, j] = False
                        elif direction == Direction.UP:
                            self.assertTrue(np.all(effect[i, :j]))
                            effect[i, :j] = False
                        elif direction == Direction.DOWN:
                            self.assertTrue(np.all(effect[i, j + 1:]))
                            effect[i, j + 1:] = False
                        else:
                            raise ValueError(direction)
                        self.assertFalse(np.any(effect))

    def test_effect2(self):
        """
        HIの利きのある4方向にそれぞれ駒をおいて利きを遮った場合のテスト

        :return:
        """
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        # 利きを遮る駒の候補を作成
        # 各方向についてテストするので、利きの長い駒以外が候補
        block_piece_list = [
            p for p in Piece if p not in (
                Piece.BLACK_KY, Piece.BLACK_KA, Piece.BLACK_HI,
                Piece.BLACK_UM, Piece.BLACK_RY, Piece.EMPTY, Piece.SIZE)
        ]

        ph = tf.placeholder(tf.int32, shape=shape)
        for direction in get_cross_directions():
            white_effect = BlackNaiveLongEffectLayer(
                direction=direction, data_format=self.data_format,
                use_cudnn=self.use_cudnn
            )(ph)
            # チャネルの処理が面倒なので、次元を下げる
            white_effect = tf.squeeze(white_effect)

            with self.test_session() as sess:
                for i, j, k, l in product(range(9), range(9), range(9),
                                          [0, 1]):
                    # KAを(i, j)に、
                    # 遮る駒を(i * l + k * (1 - l), j * (1 - l) + k * l)に置く
                    x = i * l + k * (1 - l)
                    y = j * (1 - l) + k * l
                    if i == x and j == y:
                        continue

                    with self.subTest(direction=direction, i=i, j=j, k=k, l=l):
                        board[:] = Piece.EMPTY
                        block_piece = np.random.choice(block_piece_list)
                        if self.data_format == 'NCHW':
                            board[0, 0, i, j] = Piece.BLACK_RY
                            board[0, 0, x, y] = block_piece
                        else:
                            board[0, i, j, 0] = Piece.BLACK_RY
                            board[0, x, y, 0] = block_piece

                        effect = sess.run(white_effect, feed_dict={ph: board})

                        self.assertTupleEqual(effect.shape, (9, 9))

                        # 必ず4方向のどこかにあるので、xかyを比較すれば十分
                        if direction == Direction.RIGHT:
                            block = x < i
                            u, v = slice(x, i), j
                            s, t = slice(None, i), j
                        elif direction == Direction.LEFT:
                            block = x > i
                            u, v = slice(i + 1, x + 1), j
                            s, t = slice(i + 1, None), j
                        elif direction == Direction.UP:
                            block = y < j
                            u, v = i, slice(y, j)
                            s, t = i, slice(None, j)
                        elif direction == Direction.DOWN:
                            block = y > j
                            u, v = i, slice(j + 1, y + 1)
                            s, t = i, slice(j + 1, None)
                        else:
                            raise ValueError(direction)

                        if block:
                            self.assertTrue(np.all(effect[u, v]))
                            effect[u, v] = False
                        else:
                            self.assertTrue(np.all(effect[s, t]))
                            effect[s, t] = False

                        self.assertFalse(np.any(effect))
