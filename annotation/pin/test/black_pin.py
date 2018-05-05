#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.black_effect.pseudo_ou_effect import BlackPseudoOuEffect
from annotation.direction import (Direction, get_eight_directions,
                                  get_cross_directions,
                                  get_diagonal_directions, PinDirection)
from annotation.piece import Piece
from annotation.white_naive_effect.naive_long import WhiteNaiveLongEffectLayer
from ..black_offset import BlackPinLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/06'


class TestBlackPin(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_pin1(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        ph = tf.placeholder(tf.int32, shape=shape)
        pseudo_effect = BlackPseudoOuEffect(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph)
        long_effect = {
            direction: WhiteNaiveLongEffectLayer(
                direction=direction, data_format=self.data_format,
                use_cudnn=self.use_cudnn
            )(ph) for direction in get_eight_directions()
        }
        pinned_board = BlackPinLayer(
            data_format=self.data_format
        )(ph, pseudo_effect, long_effect)
        offset = tf.squeeze(pinned_board - ph)

        long_piece_list = (Piece.WHITE_KY, Piece.WHITE_KA, Piece.WHITE_HI,
                           Piece.WHITE_UM, Piece.WHITE_RY)
        pinned_piece_list = [
            Piece(p) for p in range(Piece.BLACK_FU, Piece.WHITE_FU)
            if p != Piece.BLACK_OU
        ]

        with self.test_session() as sess:
            for tmp in product(get_eight_directions(), range(9), range(9),
                               range(1, 8)):
                direction, i, j, k = tmp

                offset_value = Piece.SIZE + PinDirection[direction.name] * 14

                # ピンされる駒の移動
                x, y = self._get_position(direction=direction, i=i, j=j, k=k)
                if x not in range(9) or y not in range(9):
                    continue

                for l in range(k + 1, 9):
                    # 利きの長い駒の位置
                    u, v = self._get_position(direction=direction,
                                              i=i, j=j, k=l)
                    if u not in range(9) or v not in range(9):
                        continue

                    board[:] = Piece.EMPTY
                    if self.data_format == 'NCHW':
                        board[0, 0, i, j] = Piece.BLACK_OU
                    else:
                        board[0, i, j, 0] = Piece.BLACK_OU

                    for long_piece in long_piece_list:
                        pinned_piece = np.random.choice(pinned_piece_list)
                        if self.data_format == 'NCHW':
                            board[0, 0, x, y] = pinned_piece
                            board[0, 0, u, v] = long_piece
                        else:
                            board[0, x, y, 0] = pinned_piece
                            board[0, u, v, 0] = long_piece

                        if long_piece == Piece.WHITE_KY:
                            pin = direction == Direction.UP
                        elif long_piece in (Piece.WHITE_KA, Piece.WHITE_UM):
                            pin = direction in get_diagonal_directions()
                        elif long_piece in (Piece.WHITE_HI, Piece.WHITE_RY):
                            pin = direction in get_cross_directions()
                        else:
                            # ここには到達しないはず
                            raise ValueError(direction)

                        with self.subTest(direction=direction, i=i, j=j, k=k,
                                          l=l, pinned_piece=pinned_piece,
                                          long_piece=long_piece):
                            o = sess.run(offset, feed_dict={ph: board})

                            if pin:
                                self.assertEqual(o[x, y], offset_value)
                                # 他がすべて0であることを確認
                                o[x, y] = 0
                                self.assertTrue(np.all(o == 0))
                            else:
                                self.assertTrue(np.all(o == 0))

    @staticmethod
    def _get_position(direction, i, j, k=1):
        table = np.array([
            [-1, -1],  # right up
            [-1, 0],  # right
            [-1, 1],  # right down
            [0, -1],  # up
            [0, 1],  # down
            [1, -1],  # left up
            [1, 0],  # left
            [1, 1],  # left down
            [-1, -2],  # right up up
            [1, -2],  # left up up
        ])

        return np.array([i, j]) + table[direction] * k
