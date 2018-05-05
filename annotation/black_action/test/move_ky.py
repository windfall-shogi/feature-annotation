#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.direction import Direction
from annotation.piece import Piece
from ..ky import BlackKyMoveLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/11'


class TestBlackKyMove(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_ky_move(self):
        """
        KYについて成り、成らずの判定のテスト
        利きが通るかどうかは別のところで判定しているので、ここでは考えない
        :return:
        """
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        # 移動距離ごとに用意
        effect = {
            Direction.UP: [np.empty(shape, dtype=np.bool) for _ in range(8)]
        }

        board = np.empty(shape, dtype=np.int32)

        ph_board = tf.placeholder(tf.int32, shape=shape)
        ky_effect = {
            Direction.UP: [
                tf.placeholder(tf.bool, shape=shape) for _ in range(8)
            ]
        }
        non_promoting, promoting = BlackKyMoveLayer(
            data_format=self.data_format
        )(ph_board, ky_effect)
        # アクセスしやすいように次元を下げる
        non_promoting = {key: tf.squeeze(value)
                         for key, value in non_promoting.items()}
        promoting = {key: tf.squeeze(value)
                     for key, value in promoting.items()}

        feed_dict = {}
        for direction, ph_list in ky_effect.items():
            for ph, e in zip(ph_list, effect[direction]):
                feed_dict[ph] = e
        feed_dict[ph_board] = board

        with self.test_session() as sess:
            for i, j, piece in product(range(9), range(9), range(Piece.SIZE)):
                for effect_list in effect.values():
                    for e in effect_list:
                        e[:] = False
                        if self.data_format == 'NCHW':
                            e[0, 0, i, j] = True
                        else:
                            e[0, i, j, 0] = True

                piece = Piece(piece)
                board[:] = piece

                n, p = sess.run([non_promoting, promoting],
                                feed_dict=feed_dict)

                b = np.squeeze(board)

                for direction, distance in product(effect.keys(), range(8)):
                    n_move = n[direction]
                    p_move = p[direction]

                    with self.subTest(i=i, j=j, piece=piece,
                                      direction=direction, distance=distance):
                        self.assertTupleEqual((8, 9, 9), n_move.shape)
                        self.assertTupleEqual((8, 9, 9), p_move.shape)

                        if b[i, j] < Piece.WHITE_FU:
                            # 自身の駒があって動けない
                            self.assertFalse(np.all(n_move[distance]))
                            self.assertFalse(np.all(p_move[distance]))
                        else:
                            if j >= 1:
                                # KYがそのままでいられるのは2段目以上
                                self.assertTrue(n_move[distance, i, j])
                                n_move[distance, i, j] = False
                            self.assertFalse(np.all(n_move[distance]))

                            if j < 3:
                                self.assertTrue(p_move[distance, i, j])
                                p_move[distance, i, j] = False
                            self.assertFalse(np.all(p_move[distance]))
