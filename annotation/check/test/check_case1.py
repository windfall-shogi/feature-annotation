#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.black_effect.pseudo_ou_effect import BlackPseudoOuEffect
from ..white_all_check import WhiteAllCheckLayer
from annotation.direction import Direction

__author__ = 'Yasuhiro'
__date__ = '2018/3/25'


class TestCheckCase1(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def test_case1(self):
        """
        動作確認中に発覚したものについて調べる
         .  n  .  .  .  .  .  n  .
         l  .  k  .  g  .  s  .  l
         .  s  .  p  .  .  .  .  .
         .  p  .  g  p  p  r  p  p
         .  .  p  .  .  R  p  P  .
         p  .  .  G  .  P  G  .  .
         P  P  P  .  .  .  .  B  P
         .  B  .  .  .  .  .  .  L
         L  N  S  .  K  .  S  N  .

         p*3

        :return:
        """
        board = np.array([[28, 15, 28, 14, 28, 28,  0,  1, 28],
                          [16, 28, 28, 14,  0, 28,  4, 28,  2],
                          [28, 17, 28, 19, 14,  6, 28, 28,  3],
                          [28, 28, 28, 14,  5,  0, 28, 28, 28],
                          [28, 20, 28, 14, 28, 28, 28, 28,  7],
                          [28, 28, 14, 20, 28,  6, 28, 28, 28],
                          [28, 21, 28, 28, 14, 28,  0, 28,  3],
                          [16, 28, 17, 14, 28, 28,  0,  4,  2],
                          [28, 15, 28, 28, 28, 14,  0, 28,  1]],
                         dtype=np.int32)
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.reshape(board, shape)

        pseudo_effect = BlackPseudoOuEffect(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(board)

        all_check, long_check = WhiteAllCheckLayer()(board, pseudo_effect)

        with self.test_session() as sess:
            all_check_, long_check_ = sess.run([all_check, long_check])

        for direction, check in all_check_.items():
            with self.subTest(direction=direction):
                self.assertFalse(check)

        for direction, check in long_check_.items():
            with self.subTest(direction=direction):
                self.assertFalse(check)
