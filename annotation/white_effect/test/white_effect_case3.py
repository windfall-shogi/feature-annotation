#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from annotation.black_naive_effect import BlackNaiveAllEffect
from annotation.direction import PinDirection, Direction, get_eight_directions
from annotation.pin import WhitePinLayer
from annotation.white_effect.pseudo_ou_effect import WhitePseudoOuEffect
from ..count import WhiteEffectCountLayer
from ..effect import WhiteEffectLayer
from ..long_effect import WhiteLongEffectLayer
from ..ou import WhiteOuEffectLayer
from ..short_effect import WhiteShortEffectLayer
from annotation.annotation import AnnotationLayer
from annotation.piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/3/28'


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

    def test_effect_case3(self):
        # 非手番側のナイーブな利きを求める
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

        # 長い利きを計算
        white_long_effect = WhiteLongEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(pinned_board)
        white_long_effect = [tf.squeeze(value) for value in white_long_effect]

        with self.test_session() as sess:
            pin = sess.run(tf.squeeze(pinned_board))
            self.assertEqual(pin[5, 3],
                             Piece.WHITE_KA + 15 + PinDirection.DIAGONAL1)
            effect = sess.run(white_long_effect)

        for direction in get_eight_directions():
            with self.subTest(direction=direction):
                if direction in (Direction.RIGHT_DOWN, Direction.LEFT_UP):
                    self.assertEqual(PinDirection.DIAGONAL2,
                                     PinDirection[direction.name])
                    # pin
                    self.assertTrue(np.all(effect[direction] == 0))
                else:
                    # 縦と横はHIが2枚あるので、それらで必ずある
                    self.assertTrue(np.any(effect[direction]))

    def test_count_case3(self):
        # 非手番側のナイーブな利きを求める
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

        # 短い利きを計算
        white_short_effect = WhiteShortEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(pinned_board)
        # 長い利きを計算
        white_long_effect = WhiteLongEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(pinned_board)
        # OUの利きを計算
        white_ou_effect = WhiteOuEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(self.board, black_all_effect)

        # マスごとの利きの個数を計算
        count = WhiteEffectCountLayer()(
            white_short_effect, white_long_effect, white_ou_effect
        )
        count = tf.squeeze(count)

        with self.test_session() as sess:
            c = sess.run(count)

        expected = np.array([[0, 0, 0, 1, 1, 0, 0, 0, 0],
                             [1, 1, 1, 0, 1, 2, 1, 0, 0],
                             [2, 0, 0, 2, 2, 0, 1, 0, 0],
                             [1, 1, 0, 1, 1, 1, 1, 0, 0],
                             [0, 1, 2, 0, 1, 1, 0, 0, 0],
                             [1, 0, 1, 2, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0, 0, 0, 1],
                             [0, 1, 0, 1, 0, 0, 0, 2, 0]])

        for i, j in product(range(9), repeat=2):
            with self.subTest(i=i, j=j):
                self.assertEqual(c[i, j], expected[i, j])

    def test_count_case3_2(self):
        count = WhiteEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(self.board)
        count = tf.squeeze(count)

        with self.test_session() as sess:
            c = sess.run(count)

        expected = np.array([[0, 0, 0, 1, 1, 0, 0, 0, 0],
                             [1, 1, 1, 0, 1, 2, 1, 0, 0],
                             [2, 0, 0, 2, 2, 0, 1, 0, 0],
                             [1, 1, 0, 1, 1, 1, 1, 0, 0],
                             [0, 1, 2, 0, 1, 1, 0, 0, 0],
                             [1, 0, 1, 2, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0, 0, 0, 1],
                             [0, 1, 0, 1, 0, 0, 0, 2, 0]])

        for i, j in product(range(9), repeat=2):
            with self.subTest(i=i, j=j):
                self.assertEqual(c[i, j], expected[i, j])

    def test_annotation(self):
        hand = np.zeros((1, 7), dtype=np.int32)
        _, _, count, _ = AnnotationLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(self.board, hand)

        count = tf.squeeze(count)

        with self.test_session() as sess:
            c = sess.run(count)

        expected = np.array([[0, 0, 0, 1, 1, 0, 0, 0, 0],
                             [1, 1, 1, 0, 1, 2, 1, 0, 0],
                             [2, 0, 0, 2, 2, 0, 1, 0, 0],
                             [1, 1, 0, 1, 1, 1, 1, 0, 0],
                             [0, 1, 2, 0, 1, 1, 0, 0, 0],
                             [1, 0, 1, 2, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0, 0, 0, 1],
                             [0, 1, 0, 1, 0, 0, 0, 2, 0]])

        for i, j in product(range(9), repeat=2):
            with self.subTest(i=i, j=j):
                self.assertEqual(c[i, j], expected[i, j])
