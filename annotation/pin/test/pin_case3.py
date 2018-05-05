#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.black_naive_effect import BlackNaiveAllEffect
from annotation.direction import PinDirection, Direction
from annotation.white_effect.pseudo_ou_effect import WhitePseudoOuEffect
from ..white_offset import WhitePinLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/27'


class TestPinCase3(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        dotenv_path = Path(__file__).parents[3] / '.env'
        load_dotenv(str(dotenv_path))

        cls.data_format = os.environ.get('DATA_FORMAT')
        cls.use_cudnn = bool(os.environ.get('USE_CUDNN'))

    def setUp(self):
        """
        動作確認中に発覚したものについて調べる
        ERROR: next effect is not match.
        [[0 0 0 1 1 0 0 0 0]
         [1 1 1 0 1 2 1 0 0]
         [2 0 0 2 2 0 1 0 0]
         [1 1 0 1 1 1 1 0 0]
         [0 1 2 0 1 1 0 0 0]
         [1 0 1 2 1 0 0 0 0]
         [1 0 0 0 1 0 0 0 0]
         [0 0 1 1 0 0 0 0 1]
         [0 1 0 1 0 0 0 2 0]]
        -----------
        [[0 0 0 1 1 0 0 0 0]
         [1 1 1 0 1 2 1 0 0]
         [2 0 0 2 2 0 1 0 0]
         [1 1 0 1 1 1 1 0 0]
         [0 1 2 0 2 1 0 0 0]
         [1 0 1 2 1 0 0 0 0]
         [1 0 1 0 1 0 0 0 0]
         [0 1 1 1 0 0 0 0 1]
         [1 1 0 1 0 0 0 2 0]]
        episode: 1, step:318
        .  .  .  .  .  .  .  . +P
        n  .  .  k  l  .  .  p  .
        p  n  .  .  p  .  p  g  P
        .  p  G  s  .  .  .  .  .
        .  P  .  p  P  p +b  .  .
        .  G  l  .  G  B  .  .  p
        .  .  S +s  .  .  .  P  R
        .  .  .  . +n  L +s  .  l
        .  .  K  .  R  . +n  .  .

        p*5

        :return:
        """
        board = np.array([[28, 28, 28, 28, 28, 28,  0,  2, 28],
                          [28, 28, 28, 20, 14,  0,  2, 28, 28],
                          [21, 28, 17,  1, 28, 20, 28, 28, 28],
                          [28, 28, 11, 28,  0,  3, 28,  7, 28],
                          [19, 10, 28, 20, 14, 28,  0,  1, 28],
                          [28, 15, 28, 18,  0, 28, 28, 28, 28],
                          [10, 11, 28, 28, 12, 28,  0, 28, 28],
                          [28, 28, 14, 28, 28, 28,  6,  0, 28],
                          [28,  1, 19,  0, 28, 28, 14, 28, 22]],
                         dtype=np.int32)
        shape = (1, 1, 9, 9) if self.data_format == 'NCHW' else (1, 9, 9, 1)
        board = np.reshape(board, shape)

        self.board = board

    def test_pin_case3(self):
        black_all_effect, black_long_effect = BlackNaiveAllEffect(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(self.board)

        pseudo_effect = WhitePseudoOuEffect(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(self.board)

        # ピンされているかを判定
        pinned_board = WhitePinLayer(
            data_format=self.data_format
        )(self.board, pseudo_effect, black_long_effect)
        pinned_board = tf.squeeze(pinned_board)

        with self.test_session() as sess:
            pin = sess.run(pinned_board)

        board = np.squeeze(self.board)
        for i, j in product(range(9), repeat=2):
            with self.subTest(i=i, j=j):
                if i == 5 and j == 3:
                    self.assertEqual(
                        pin[i, j],
                        board[i, j] + 15 + 14 * PinDirection.DIAGONAL1
                    )
                    self.assertEqual(PinDirection.DIAGONAL1,
                                     PinDirection[Direction.RIGHT_UP.name])
                elif i == 2 and j == 2:
                    self.assertEqual(
                        pin[i, j],
                        board[i, j] + 15 + 14 * PinDirection.VERTICAL
                    )
                else:
                    self.assertEqual(pin[i, j], board[i, j])
