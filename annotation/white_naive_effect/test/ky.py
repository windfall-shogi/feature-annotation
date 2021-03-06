#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import unittest
from itertools import product, chain

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.piece import Piece
from annotation.direction import Direction, get_eight_directions

from ..naive_long import WhiteNaiveLongEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/2/28'


class TestWhiteLongEffectKy(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_effect1(self):
        """
        KYの利きがあるかを確認するテスト
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
                # 9段目にはKYを置けないけど、気にしない
                for i, j in product(range(9), repeat=2):
                    with self.subTest(direction=direction, i=i, j=j):
                        board[:] = Piece.EMPTY
                        if self.data_format == 'NCHW':
                            board[0, 0, i, j] = Piece.WHITE_KY
                        else:
                            board[0, i, j, 0] = Piece.WHITE_KY

                        effect = sess.run(white_effect, feed_dict={ph: board})

                        self.assertTupleEqual(effect.shape, (9, 9))

                        if direction != Direction.DOWN:
                            # 利きがあるマスはない
                            self.assertFalse(np.any(effect))
                            continue

                        if j == 8:
                            # 盤の端に駒があるので、盤の中に利きはない
                            self.assertFalse(np.any(effect))
                        else:
                            self.assertTrue(np.all(effect[i, j + 1:]))
                            # 利きがあるはずの場所の利きを消す
                            effect[i, j + 1:] = False
                            self.assertFalse(np.any(effect))

    def test_effect2(self):
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

            # 利きを遮る駒の候補を作成
            # 各方向についてテストするので、利きの長い駒以外が候補
            block_piece_list = [
                p for p in Piece if p not in (Piece.WHITE_KY, Piece.WHITE_KA,
                                              Piece.WHITE_HI, Piece.WHITE_UM,
                                              Piece.WHITE_RY, Piece.EMPTY,
                                              Piece.SIZE)
            ]

            with self.test_session() as sess:
                # 9段目にはKYを置けないけど、気にしない
                for i, j, k in product(range(9), repeat=3):
                    # KYを(i, j)に、遮る駒を(i, k)に置く
                    if j == k:
                        continue

                    with self.subTest(direction=direction, i=i, j=j, k=k):
                        board[:] = Piece.EMPTY
                        block_piece = np.random.choice(block_piece_list)
                        if self.data_format == 'NCHW':
                            board[0, 0, i, j] = Piece.WHITE_KY
                            board[0, 0, i, k] = block_piece
                        else:
                            board[0, i, j, 0] = Piece.WHITE_KY
                            board[0, i, k, 0] = block_piece

                        effect = sess.run(white_effect, feed_dict={ph: board})

                        self.assertTupleEqual(effect.shape, (9, 9))

                        if direction != Direction.DOWN:
                            # 利きがあるマスはない
                            self.assertFalse(np.any(effect), msg=block_piece)
                            continue

                        if j == 8:
                            # 盤の端に駒があるので、盤の中に利きはない
                            self.assertFalse(np.any(effect))
                        else:
                            if j < k:
                                # 駒が利きを遮っている
                                self.assertTrue(np.all(effect[i, j + 1:k + 1]))
                                # 利きがあるはずの場所の利きを消す
                                effect[i, j + 1:k + 1] = False
                            else:
                                # KYの後ろにあるので、遮っていない
                                self.assertTrue(np.all(effect[i, j + 1:]))
                                # 利きがあるはずの場所の利きを消す
                                effect[i, j + 1:] = False
                            self.assertFalse(np.any(effect), msg=effect)
