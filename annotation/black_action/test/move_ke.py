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
from ..ke import BlackKeMoveLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/12'


class TestBlackKeMove(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_ke_move(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        effect = {direction: np.empty(shape, dtype=np.bool)
                  for direction in (Direction.RIGHT_UP_UP,
                                    Direction.LEFT_UP_UP)}

        board = np.empty(shape, dtype=np.int32)

        ph_board = tf.placeholder(tf.int32, shape=shape)
        ke_effect = {direction: tf.placeholder(tf.bool, shape=shape)
                     for direction in effect.keys()}
        non_promoting, promoting = BlackKeMoveLayer(
            data_format=self.data_format
        )(ph_board, ke_effect)
        # アクセスしやすいように次元を下げる
        non_promoting = {key: tf.squeeze(value)
                         for key, value in non_promoting.items()}
        promoting = {key: tf.squeeze(value)
                     for key, value in promoting.items()}

        feed_dict = {placeholder: effect[direction]
                     for direction, placeholder in ke_effect.items()}
        feed_dict[ph_board] = board

        with self.test_session() as sess:
            for i, j, piece in product(range(9), range(9), range(Piece.SIZE)):
                for e in effect.values():
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

                for direction in effect.keys():
                    n_move = n[direction]
                    p_move = p[direction]

                    with self.subTest(i=i, j=j, direction=direction):
                        self.assertTupleEqual((9, 9), n_move.shape)
                        self.assertTupleEqual((9, 9), p_move.shape)

                        if b[i, j] < Piece.WHITE_FU:
                            # 自身の駒があって動けない
                            self.assertFalse(np.all(n_move))
                            self.assertFalse(np.all(p_move))
                        else:
                            if j >= 2:
                                # KEがそのままでいられるのは3段目以上
                                self.assertTrue(n_move[i, j])
                                n_move[i, j] = False
                            self.assertFalse(np.all(n_move))

                            if j < 3:
                                self.assertTrue(p_move[i, j])
                                p_move[i, j] = False
                            self.assertFalse(np.all(p_move))
