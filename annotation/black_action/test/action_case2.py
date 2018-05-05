#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.black_effect import BlackEffectLayer
from ..action import BlackActionLayer
from annotation.direction import Direction

__author__ = 'Yasuhiro'
__date__ = '2018/3/26'


class TestActionCase2(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def setUp(self):
        """
        動作確認中に発覚したものについて調べる
        ERROR: the number of actions is not match.
        3 v.s. 35
        episode: 0, step:113
         .  B  k  .  .  .  .  .  .
         .  .  .  .  .  g  s  p  l
         .  .  p  p  .  .  .  .  n
         s  .  .  g  .  p  .  P  .
         G  p  .  .  p  .  .  r  p
         p  P  P  P  .  P  p  .  .
         .  .  .  .  .  K  N  G  P
        +p  B  .  .  .  . +r  S  L
         L  N  S  .  L  .  .  .  .

         N*1 p*2

        :return:
        """
        board = np.array([[28, 15, 16, 28, 14, 28, 0, 1, 28],
                          [28, 14, 28, 0, 19, 28, 6, 3, 28],
                          [28, 17, 28, 28, 28, 14, 2, 27, 28],
                          [28, 20, 28, 14, 28, 0, 7, 28, 28],
                          [28, 28, 28, 28, 14, 28, 28, 28, 1],
                          [28, 28, 14, 20, 28, 0, 28, 28, 28],
                          [21, 28, 14, 28, 28, 0, 28, 28, 3],
                          [4, 28, 28, 28, 14, 0, 28, 4, 2],
                          [28, 28, 28, 17, 6, 14, 28, 22, 1]],
                         dtype=np.int32)
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.reshape(board, shape)

        self.board = board
        self.black_hand = np.array([[0, 0, 1, 0, 0, 0, 0]], dtype=np.int32)

    def test_action_case2(self):
        (black_all_effects, black_count, black_check,
         available_square) = BlackEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(self.board)
        all_actions = BlackActionLayer(
            data_format=self.data_format
        )(self.board, self.black_hand, black_all_effects, available_square)

        with self.test_session() as sess:
            actions = sess.run(all_actions)

        for distance, direction, promotion in product(range(8), range(8),
                                                      range(2)):
            index = distance * 16 + promotion * 8 + direction
            with self.subTest(distance=distance, direction=direction,
                              promotion=promotion):
                if (distance == 0 and promotion == 0 and
                        direction in (Direction.RIGHT_UP, Direction.RIGHT_DOWN,
                                      Direction.LEFT)):
                    self.assertEqual(np.sum(actions[index]), 1)
                else:
                    self.assertTrue(np.all(actions[index] == 0))
        # KE
        for i in range(4):
            with self.subTest(i=i):
                self.assertTrue(np.all(actions[128 + i] == 0))
        # drop
        for i in range(7):
            with self.subTest(i=i):
                self.assertTrue(np.all(actions[132 + i] == 0))
