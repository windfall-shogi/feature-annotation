#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product, chain
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.direction import Direction, get_eight_directions
from annotation.piece import Piece
from ..naive_short import WhiteNaiveShortEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/01'


class TestWhiteShortEffectGi(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_effect(self):
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.empty(shape, dtype=np.int32)

        gi_directions = (Direction.RIGHT_UP, Direction.RIGHT_DOWN,
                         Direction.DOWN, Direction.LEFT_UP,
                         Direction.LEFT_DOWN)

        ph = tf.placeholder(tf.int32, shape=shape)
        for direction in chain(get_eight_directions(),
                               [Direction.RIGHT_DOWN_DOWN,
                                Direction.LEFT_DOWN_DOWN]):
            white_effect = WhiteNaiveShortEffectLayer(
                direction=direction, data_format=self.data_format,
                use_cudnn=self.use_cudnn
            )(ph)
            # チャネルの処理が面倒なので、次元を下げる
            white_effect = tf.squeeze(white_effect)

            with self.test_session() as sess:
                for i, j in product(range(9), repeat=2):
                    with self.subTest(direction=direction, i=i, j=j):
                        board[:] = Piece.EMPTY
                        if self.data_format == 'NCHW':
                            board[0, 0, i, j] = Piece.WHITE_GI
                        else:
                            board[0, i, j, 0] = Piece.WHITE_GI

                        effect = sess.run(white_effect, feed_dict={ph: board})

                        self.assertTupleEqual(effect.shape, (9, 9))

                        if direction not in gi_directions:
                            # 利きがあるマスはない
                            self.assertFalse(np.any(effect))
                            continue

                        if direction == Direction.RIGHT_UP:
                            u, v = i - 1, j - 1
                        elif direction == Direction.RIGHT_DOWN:
                            u, v = i - 1, j + 1
                        elif direction == Direction.DOWN:
                            u, v = i, j + 1
                        elif direction == Direction.LEFT_UP:
                            u, v = i + 1, j - 1
                        elif direction == Direction.LEFT_DOWN:
                            u, v = i + 1, j + 1
                        else:
                            # ここには到達しないはず
                            raise ValueError()
                        if u not in range(9) or v not in range(9):
                            # 盤の端に駒があるので、盤の中に利きはない
                            self.assertFalse(np.any(effect))
                            continue

                        self.assertTrue(effect[u, v])
                        # 利きがあるのは1か所だけのはず
                        effect[u, v] = False
                        self.assertFalse(np.any(effect))
