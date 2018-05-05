#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.piece import Piece
from ..ke import BlackKeDropLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/12'


class TestBlackKeDrop(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_ke_drop(self):
        """
        空いているマスに駒を打てるかを判定する
        :return:
        """
        black_hand = np.zeros((1, 7), dtype=np.int32)

        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)

        # 駒の配置をランダムに設定
        board = np.random.randint(Piece.SIZE, size=shape)
        # 一段目と二段目には打てないことを確認するために空ける
        if self.data_format == 'NCHW':
            board[0, 0, :, :2] = Piece.EMPTY
        else:
            board[0, :, :2, 0] = Piece.EMPTY

        ph = tf.placeholder(tf.int32, shape=black_hand.shape)
        drop_square = BlackKeDropLayer(data_format=self.data_format)(board, ph)
        # アクセスしやすいように次元を下げる
        drop_square = tf.squeeze(drop_square)

        # レイヤーに渡したのでもう変更しても大丈夫
        board = np.squeeze(board)

        with self.test_session() as sess:
            # 持ち駒の全ての枚数を試す
            for n in range(5):
                black_hand[0, Piece.BLACK_KE] = n
                square = sess.run(drop_square, feed_dict={ph: black_hand})

                with self.subTest(n=n):
                    self.assertTupleEqual((9, 9), square.shape)

                    if n == 0:
                        self.assertFalse(np.all(square))
                    else:
                        for i, j in product(range(9), repeat=2):
                            if j < 2:
                                # 一段目と二段目には打てない
                                self.assertFalse(square[i, j])
                            else:
                                # 空いているなら打てる
                                self.assertEqual(board[i, j] == Piece.EMPTY,
                                                 square[i, j])
