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
from ..short_effect import WhiteShortEffectLayer
from ..long_effect import WhiteLongEffectLayer
from ..count import WhiteEffectCountLayer
from ..ou import WhiteOuEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/22'


class TestOuEffect(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_ou_effect(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        # 適当なマスクを設定
        black_effect_mask = np.zeros(81, dtype=np.bool)
        black_effect_mask[::2] = True
        black_effect_mask = np.reshape(black_effect_mask, shape)

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

        # Layerに渡したので、変更しても大丈夫
        # アクセスしやすいように次元を下げる
        black_effect_mask = np.squeeze(black_effect_mask)

        with self.test_session() as sess:
            for i, j in product(range(9), repeat=2):
                # (i, j)に駒を置く

                board[:] = Piece.EMPTY
                if self.data_format == 'NCHW':
                    board[0, 0, i, j] = Piece.WHITE_OU
                else:
                    board[0, i, j, 0] = Piece.WHITE_OU

                count = sess.run(effect_count, feed_dict={ph: board})
                with self.subTest(i=i, j=j):
                    c = 0
                    for x, y in ((i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                                 (i, j - 1), (i, j + 1),
                                 (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)):
                        if (x in range(9) and y in range(9) and
                                not black_effect_mask[x, y]):
                            self.assertEqual(count[x, y], 1)
                            c += 1
                    self.assertEqual(np.sum(count), c)
