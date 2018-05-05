#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.direction import (Direction, get_diagonal_directions,
                                  get_cross_directions)
from annotation.piece import Piece
from ..ry import BlackRyMoveLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/14'


class TestBlackRyMove(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_ry_move(self):
        """
        RYについて成り、成らずの判定のテスト
        利きが通るかどうかは別のところで判定しているので、ここでは考えない
        :return:
        """
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        # 移動距離ごとに用意
        effect = {
            direction: [np.empty(shape, dtype=np.bool) for _ in range(8)]
            for direction in get_cross_directions()
        }
        effect.update({
            direction: np.empty(shape, dtype=np.bool)
            for direction in get_diagonal_directions()
        })

        board = np.empty(shape, dtype=np.int32)

        ph_board = tf.placeholder(tf.int32, shape=shape)
        ry_effect = {
            direction: [
                tf.placeholder(tf.bool, shape=shape) for _ in range(8)
            ] for direction in get_cross_directions()
        }
        ry_effect.update({
            direction: tf.placeholder(tf.bool, shape=shape)
            for direction in get_diagonal_directions()
        })
        non_promoting = BlackRyMoveLayer()(ph_board, ry_effect)
        # アクセスしやすいように次元を下げる
        non_promoting = {key: tf.squeeze(value)
                         for key, value in non_promoting.items()}

        feed_dict = {}
        for direction, ph_list in ry_effect.items():
            if direction in get_cross_directions():
                for ph, e in zip(ph_list, effect[direction]):
                    feed_dict[ph] = e
            else:
                feed_dict[ph_list] = effect[direction]
        feed_dict[ph_board] = board

        with self.test_session() as sess:
            for i, j, piece in product(range(9), range(9), range(Piece.SIZE)):
                for direction, effect_list in effect.items():
                    if direction in get_cross_directions():
                        for e in effect_list:
                            e[:] = False
                            if self.data_format == 'NCHW':
                                e[0, 0, i, j] = True
                            else:
                                e[0, i, j, 0] = True
                    else:
                        effect_list[:] = False
                        if self.data_format == 'NCHW':
                            effect_list[0, 0, i, j] = True
                        else:
                            effect_list[0, i, j, 0] = True

                piece = Piece(piece)
                board[:] = piece

                n = sess.run(non_promoting, feed_dict=feed_dict)

                b = np.squeeze(board)

                for direction, distance in product(effect.keys(), range(8)):
                    if direction in get_diagonal_directions():
                        if distance > 0:
                            continue

                        if (direction == Direction.RIGHT_UP and
                                (i == 8 or j == 8)):
                            continue
                        elif (direction == Direction.RIGHT_DOWN and
                                (i == 8 or j == 0)):
                            continue
                        elif (direction == Direction.LEFT_UP and
                                (i == 8 or j == 8)):
                            continue
                        elif (direction == Direction.LEFT_DOWN and
                                (i == 0 or j == 0)):
                            continue

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

                    if direction in get_cross_directions():
                        with self.subTest(i=i, j=j, piece=piece,
                                          direction=direction,
                                          distance=distance):
                            self.assertTupleEqual((8, 9, 9), n_move.shape)

                            if b[i, j] < Piece.WHITE_FU:
                                # 自身の駒があって動けない
                                self.assertFalse(np.all(n_move[distance]))
                            else:
                                self.assertTrue(n_move[distance, i, j])
                                n_move[distance, i, j] = False
                                self.assertFalse(np.all(n_move[distance]))
                    else:
                        with self.subTest(i=i, j=j, piece=piece,
                                          direction=direction):
                            self.assertTupleEqual((9, 9), n_move.shape)

                            if b[i, j] < Piece.WHITE_FU:
                                # 自身の駒があって動けない
                                self.assertFalse(np.all(n_move))
                            else:
                                self.assertTrue(n_move[i, j])
                                n_move[i, j] = False
                                self.assertFalse(np.all(n_move))
