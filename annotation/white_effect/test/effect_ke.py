#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.direction import PinDirection
from annotation.piece import Piece
from ..count import WhiteEffectCountLayer
from ..long_effect import WhiteLongEffectLayer
from ..ou import WhiteOuEffectLayer
from ..short_effect import WhiteShortEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/21'


class TestKeEffect(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_ke_effect(self):
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

        with self.test_session() as sess:
            for i, j in product(range(9), repeat=2):
                # (i, j)に駒を置く、(i + 1, j + 2)か(i - 1, j + 2)が利きのある位置
                # iについては少なくとも一方は利きがあるので、スキップの条件に含めない
                if j > 6:
                    continue

                board[:] = Piece.EMPTY
                if self.data_format == 'NCHW':
                    board[0, 0, i, j] = Piece.WHITE_KE
                else:
                    board[0, i, j, 0] = Piece.WHITE_KE

                effect = sess.run(effect_count, feed_dict={ph: board})
                with self.subTest(i=i, j=j):
                    c = 0
                    for x, y in ((i - 1, j + 2), (i + 1, j + 2)):
                        if x in range(9):
                            self.assertEqual(effect[x, y], 1)
                            c += 1
                    self.assertEqual(np.sum(effect), c)

    def test_ke_pin(self):
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

        with self.test_session() as sess:
            for i, j in product(range(9), repeat=2):
                # (i, j)に駒を置く、(i + 1, j + 2)か(i - 1, j + 2)が利きのある位置
                # iについては少なくとも一方は利きがあるので、スキップの条件に含めない
                if j > 6:
                    continue

                board[:] = Piece.EMPTY
                for pin_direction in PinDirection:
                    if pin_direction == PinDirection.SIZE:
                        continue

                    offset = Piece.SIZE - Piece.WHITE_FU + pin_direction * 14
                    if self.data_format == 'NCHW':
                        board[0, 0, i, j] = Piece.WHITE_KE + offset
                    else:
                        board[0, i, j, 0] = Piece.WHITE_KE + offset

                    count = sess.run(effect_count, feed_dict={ph: board})
                    with self.subTest(i=i, j=j, pin_direction=pin_direction):
                        self.assertTrue(np.all(count == 0))
