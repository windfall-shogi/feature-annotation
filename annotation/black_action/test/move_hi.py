#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.direction import Direction, get_cross_directions
from annotation.piece import Piece
from ..hi import BlackHiMoveLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/13'


class TestBlackHiMove(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_hi_move(self):
        """
        HIについて成り、成らずの判定のテスト
        利きが通るかどうかは別のところで判定しているので、ここでは考えない
        :return:
        """
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        # 移動距離ごとに用意
        effect = {
            direction: [np.empty(shape, dtype=np.bool) for _ in range(8)]
            for direction in get_cross_directions()
        }

        board = np.empty(shape, dtype=np.int32)

        ph_board = tf.placeholder(tf.int32, shape=shape)
        ka_effect = {
            direction: [
                tf.placeholder(tf.bool, shape=shape) for _ in range(8)
            ] for direction in effect.keys()
        }
        non_promoting, promoting = BlackHiMoveLayer(
            data_format=self.data_format
        )(ph_board, ka_effect)
        # アクセスしやすいように次元を下げる
        non_promoting = {key: tf.squeeze(value)
                         for key, value in non_promoting.items()}
        promoting = {key: tf.squeeze(value)
                     for key, value in promoting.items()}

        feed_dict = {}
        for direction, ph_list in ka_effect.items():
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
                    if direction == Direction.RIGHT:
                        if i + distance >= 8:
                            continue
                    elif direction == Direction.UP:
                        if j + distance >= 8:
                            continue
                    elif direction == Direction.DOWN:
                        if j - distance <= 0:
                            continue
                    elif direction == Direction.LEFT:
                        if i - distance <= 0:
                            continue

                    n_move = n[direction]
                    p_move = p[direction]

                    if j < 3:
                        # 方向に依らず成れる
                        promotable = True
                    elif direction == Direction.DOWN:
                        promotable = distance + 1 < j < distance + 4
                    else:
                        promotable = False

                    with self.subTest(i=i, j=j, piece=piece,
                                      direction=direction, distance=distance):
                        self.assertTupleEqual((8, 9, 9), n_move.shape)
                        self.assertTupleEqual((8, 9, 9), p_move.shape)

                        if b[i, j] < Piece.WHITE_FU:
                            # 自身の駒があって動けない
                            self.assertFalse(np.all(n_move[distance]))
                            self.assertFalse(np.all(p_move[distance]))
                        else:
                            self.assertTrue(n_move[distance, i, j])
                            n_move[distance, i, j] = False
                            self.assertFalse(np.all(n_move[distance]))

                            if promotable:
                                self.assertTrue(p_move[distance, i, j])
                                p_move[distance, i, j] = False
                            self.assertFalse(np.all(p_move[distance]))
