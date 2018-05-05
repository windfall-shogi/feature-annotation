#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.direction import Direction, PinDirection
from annotation.piece import Piece
from ..count import WhiteEffectCountLayer
from ..long_effect import WhiteLongEffectLayer
from ..ou import WhiteOuEffectLayer
from ..short_effect import WhiteShortEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/22'


class TestKiEffect(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_ki_effect(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        # ここでは相手の利きがない設定
        black_effect_mask = np.zeros(shape, dtype=np.bool)

        ph = tf.placeholder(tf.int32, shape=shape)
        short_effect = WhiteShortEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph)
        long_effect = WhiteLongEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph)
        ou_effect = WhiteOuEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph, black_effect_mask)
        effect_count = WhiteEffectCountLayer()(
            short_effect, long_effect, ou_effect
        )
        effect_count = tf.squeeze(effect_count)

        ki_pieces = [Piece.WHITE_KI, Piece.WHITE_TO, Piece.WHITE_NY,
                     Piece.WHITE_NK, Piece.WHITE_NG]

        with self.test_session() as sess:
            for i, j, piece in product(range(9), range(9), ki_pieces):
                # (i, j)に駒を置く

                board[:] = Piece.EMPTY
                if self.data_format == 'NCHW':
                    board[0, 0, i, j] = piece
                else:
                    board[0, i, j, 0] = piece

                count = sess.run(effect_count, feed_dict={ph: board})
                with self.subTest(i=i, j=j, piece=piece):
                    c = 0
                    for x, y in ((i - 1, j), (i - 1, j + 1),
                                 (i, j - 1), (i, j + 1),
                                 (i + 1, j), (i + 1, j + 1)):
                        if x in range(9) and y in range(9):
                            self.assertEqual(count[x, y], 1)
                            c += 1
                    self.assertEqual(np.sum(count), c)

    def test_ki_pin(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        # ここでは相手の利きがない設定
        black_effect_mask = np.zeros(shape, dtype=np.bool)

        ph = tf.placeholder(tf.int32, shape=shape)
        short_effect = WhiteShortEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph)
        long_effect = WhiteLongEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph)
        ou_effect = WhiteOuEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(ph, black_effect_mask)
        effect_count = WhiteEffectCountLayer()(
            short_effect, long_effect, ou_effect
        )
        effect_count = tf.squeeze(effect_count)

        ki_pieces = [Piece.BLACK_KI, Piece.BLACK_TO, Piece.BLACK_NY,
                     Piece.BLACK_NK, Piece.BLACK_NG]

        with self.test_session() as sess:
            for i, j, piece in product(range(9), range(9), ki_pieces):
                # (i, j)に駒を置く

                board[:] = Piece.EMPTY
                for pin_direction in PinDirection:
                    if pin_direction == PinDirection.SIZE:
                        continue

                    offset = Piece.SIZE + pin_direction * 14
                    if self.data_format == 'NCHW':
                        board[0, 0, i, j] = piece + offset
                    else:
                        board[0, i, j, 0] = piece + offset

                    count = sess.run(effect_count, feed_dict={ph: board})
                    with self.subTest(i=i, j=j, pin_direction=pin_direction):
                        c = 0

                        if pin_direction == PinDirection.VERTICAL:
                            square_list = ((i, j - 1), (i, j + 1))
                        elif pin_direction == PinDirection.HORIZONTAL:
                            square_list = ((i - 1, j), (i + 1, j))
                        elif pin_direction == PinDirection.DIAGONAL1:
                            square_list = ((i + 1, j + 1),)
                        elif pin_direction == PinDirection.DIAGONAL2:
                            square_list = ((i - 1, j + 1),)
                        else:
                            raise ValueError(pin_direction)

                        for x, y in square_list:
                            if x in range(9) and y in range(9):
                                self.assertEqual(count[x, y], 1)
                                c += 1
                        self.assertEqual(np.sum(count), c)

    @staticmethod
    def _get_square(i, j, direction):
        if direction == Direction.RIGHT_UP:
            x = i - 1
            y = j - 1
        elif direction == Direction.RIGHT:
            x = i - 1
            y = j
        elif direction == Direction.UP:
            x = i
            y = j - 1
        elif direction == Direction.DOWN:
            x = i
            y = j + 1
        elif direction == Direction.LEFT_UP:
            x = i + 1
            y = j - 1
        elif direction == Direction.LEFT:
            x = i + 1
            y = j
        else:
            raise ValueError(direction)

        return x, y
