#!/usr/bin/env python3
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
                                  get_diagonal_directions)

from ..naive_long import WhiteNaiveLongEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/01'


class TestWhiteLongEffectKa(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_effect1(self):
        """
        KAの利きがあるかを確認するテスト
        他の駒が利きを遮る場合は考えない

        :return:
        """
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        ph = tf.placeholder(tf.int32, shape=shape)
        for direction in chain(get_eight_directions(),
                               [Direction.RIGHT_DOWN_DOWN,
                                Direction.LEFT_DOWN_DOWN]):
            if direction in (Direction.RIGHT_DOWN_DOWN,
                             Direction.LEFT_DOWN_DOWN):
                # 桂馬の方向の長い利きはあり得ないのでエラー
                with self.assertRaises(ValueError):
                    WhiteNaiveLongEffectLayer(
                        direction=direction, data_format=self.data_format,
                        use_cudnn=self.use_cudnn
                    )(ph)
                continue

            white_effect = WhiteNaiveLongEffectLayer(
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
                            board[0, 0, i, j] = Piece.WHITE_KA
                        else:
                            board[0, i, j, 0] = Piece.WHITE_KA

                        effect = sess.run(white_effect, feed_dict={ph: board})

                        self.assertTupleEqual(effect.shape, (9, 9))

                        if direction not in get_diagonal_directions():
                            # 利きがあるマスはない
                            self.assertFalse(np.any(effect))
                            continue

                        right, up = self.get_direction_flag(
                            direction=direction
                        )
                        edge1, op1 = self.get_edge_flag(n=i, flag=right)
                        edge2, op2 = self.get_edge_flag(n=j, flag=up)

                        if edge1 or edge2:
                            # 盤の端に駒があるので、盤の中に利きはない
                            self.assertFalse(np.any(effect))
                            continue

                        margin = min(i if right else 8 - i,
                                     j if up else 8 - j)
                        for n in range(1, margin + 1):
                            u, v = op1(i, n), op2(j, n)
                            self.assertTrue(effect[u, v])
                            effect[u, v] = False
                        self.assertFalse(np.any(effect))

    def test_effect2(self):
        """
        KAの利きのある4方向にそれぞれ駒をおいて利きを遮った場合のテスト

        :return:
        """
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        # 利きを遮る駒の候補を作成
        # 各方向についてテストするので、利きの長い駒以外が候補
        block_piece_list = [
            p for p in Piece if p not in (
                Piece.WHITE_KY, Piece.WHITE_KA, Piece.WHITE_HI,
                Piece.WHITE_UM, Piece.WHITE_RY, Piece.EMPTY, Piece.SIZE
            )
        ]

        ph = tf.placeholder(tf.int32, shape=shape)
        for direction in get_diagonal_directions():
            white_effect = WhiteNaiveLongEffectLayer(
                direction=direction, data_format=self.data_format,
                use_cudnn=self.use_cudnn
            )(ph)
            # チャネルの処理が面倒なので、次元を下げる
            white_effect = tf.squeeze(white_effect)

            with self.test_session() as sess:
                for i, j, k, l in product(range(9), range(9), range(9),
                                          [-1, 1]):
                    # KAを(i, j)に、遮る駒を(x, k)に置く
                    if j == k:
                        continue

                    # 利きを遮る駒を置く位置を計算
                    x = i + (k - j) * l
                    if x < 0 or x >= 9:
                        continue

                    with self.subTest(direction=direction, i=i, j=j, k=k):
                        board[:] = Piece.EMPTY
                        block_piece = np.random.choice(block_piece_list)
                        if self.data_format == 'NCHW':
                            board[0, 0, i, j] = Piece.WHITE_KA
                            board[0, 0, x, k] = block_piece
                        else:
                            board[0, i, j, 0] = Piece.WHITE_KA
                            board[0, x, k, 0] = block_piece

                        effect = sess.run(white_effect, feed_dict={ph: board})

                        self.assertTupleEqual(effect.shape, (9, 9))

                        right, up = self.get_direction_flag(
                            direction=direction
                        )
                        edge1, op1 = self.get_edge_flag(n=i, flag=right)
                        edge2, op2 = self.get_edge_flag(n=j, flag=up)

                        if edge1 or edge2:
                            # 盤の端に駒があるので、盤の中に利きはない
                            self.assertFalse(np.any(effect))
                            continue

                        if k - j < 0:
                            if l > 0:
                                block = direction == Direction.RIGHT_UP
                            else:
                                block = direction == Direction.LEFT_UP
                        else:
                            if l > 0:
                                block = direction == Direction.LEFT_DOWN
                            else:
                                block = direction == Direction.RIGHT_DOWN

                        if block:
                            # (i, j)から(x, k)までの距離を計算
                            margin = min(i - x if right else x - i,
                                         j - k if up else k - j)
                        else:
                            # 盤の端まで利きが伸びている
                            margin = min(i if right else 8 - i,
                                         j if up else 8 - j)
                        for n in range(1, margin + 1):
                            u, v = op1(i, n), op2(j, n)
                            self.assertTrue(effect[u, v])
                            effect[u, v] = False
                        self.assertFalse(np.any(effect))

    @staticmethod
    def get_direction_flag(direction):
        if direction == Direction.RIGHT_UP:
            right = True
            up = True
        elif direction == Direction.RIGHT_DOWN:
            right = True
            up = False
        elif direction == Direction.LEFT_UP:
            right = False
            up = True
        elif direction == Direction.LEFT_DOWN:
            right = False
            up = False
        else:
            # ここには到達しないはず
            raise ValueError()
        return right, up

    @staticmethod
    def get_edge_flag(n, flag):
        if flag:
            edge = n == 0
            op = sub
        else:
            edge = n == 8
            op = add

        return edge, op
