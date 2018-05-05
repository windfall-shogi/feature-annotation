#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from itertools import product, chain
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.direction import Direction, get_eight_directions
from annotation.piece import Piece
from ..ou import BlackOuMoveLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/14'


class TestBlackKiMove(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_ki_move(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        effect = {
            direction: np.empty(shape, dtype=np.bool)
            for direction in get_eight_directions()
        }

        board = np.empty(shape, dtype=np.int32)

        ph_board = tf.placeholder(tf.int32, shape=shape)
        ki_effect = {direction: tf.placeholder(tf.bool, shape=shape)
                     for direction in effect.keys()}
        non_promoting = BlackOuMoveLayer()(ph_board, ki_effect)
        # アクセスしやすいように次元を下げる
        non_promoting = {key: tf.squeeze(value)
                         for key, value in non_promoting.items()}

        feed_dict = {placeholder: effect[direction]
                     for direction, placeholder in ki_effect.items()}
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

                n = sess.run(non_promoting, feed_dict=feed_dict)

                b = np.squeeze(board)

                for direction, move in n.items():
                    with self.subTest(i=i, j=j, direction=direction):
                        self.assertTupleEqual((9, 9), move.shape)

                        if b[i, j] < Piece.WHITE_FU:
                            # 自身の駒があって動けない
                            self.assertFalse(np.all(move))
                        else:
                            self.assertTrue(move[i, j])
                            move[i, j] = False
                            self.assertFalse(np.all(move))
