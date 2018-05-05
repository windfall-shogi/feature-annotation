#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.black_effect.pseudo_ou_effect import BlackPseudoOuEffect
from ..white_all_check import WhiteAllCheckLayer
from annotation.direction import Direction
from ..black_available_square import CheckAvailableSquareLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/25'


class TestCheckCase1(tf.test.TestCase):
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

    def test_case2_check(self):
        """
        王手が正しく判定されているかを確認

        :return:
        """
        pseudo_effect = BlackPseudoOuEffect(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(self.board)

        all_check, long_check = WhiteAllCheckLayer()(self.board, pseudo_effect)

        with self.test_session() as sess:
            all_check_, long_check_ = sess.run([all_check, long_check])

        for direction, check in all_check_.items():
            with self.subTest(direction=direction):
                self.assertEqual(direction == Direction.RIGHT_DOWN, check)

        for direction, check in long_check_.items():
            with self.subTest(direction=direction):
                self.assertFalse(check)

    def test_available_square(self):
        pseudo_effect = BlackPseudoOuEffect(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(self.board)

        all_check, long_check = WhiteAllCheckLayer()(self.board, pseudo_effect)

        available_square = CheckAvailableSquareLayer()(pseudo_effect,
                                                       all_check)

        available_square = tf.squeeze(available_square)
        with self.test_session() as sess:
            square = sess.run(available_square)

        for i, j in product(range(9), repeat=2):
            # 3,8にある龍を捕る
            available = i == 2 and j == 7
            with self.subTest(i=i, j=j):
                self.assertEqual(square[i, j], available)
