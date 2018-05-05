#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from ..white_naive_effect import WhiteNaiveAllEffect
from .pseudo_ou_effect import BlackPseudoOuEffect
from ..pin import BlackPinLayer
from ..check import WhiteAllCheckLayer, CheckAvailableSquareLayer
from .all_pieces import BlackAllEffectLayer
from .count import BlackEffectCountLayer

__author__ = 'Yasuhiro'
__date__ = '2018/2/18'


class BlackEffectLayer(snt.AbstractModule):
    def __init__(self, data_format, use_cudnn=True, name='black_effect'):
        super().__init__(name=name)
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, board):
        """
        先後のそれぞれの利きと手番側の合法手を求める
        合法手を求める過程で王手の有無、ピンを検出する

        :param board:
        :return:
        """
        # 非手番側の利きを求める
        white_all_effect, white_long_effect = WhiteNaiveAllEffect(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(board)

        pseudo_effect = BlackPseudoOuEffect(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(board)
        # 王手を判定
        all_check, long_check = WhiteAllCheckLayer()(board, pseudo_effect)
        # 王手を考慮した移動可能な領域の候補
        available_square = CheckAvailableSquareLayer()(pseudo_effect,
                                                       all_check)
        # ピンされているかを判定
        pinned_board = BlackPinLayer(
            data_format=self.data_format
        )(board, pseudo_effect.long, white_long_effect)
        # 駒ごとに利きを計算
        all_effects = BlackAllEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(pinned_board, available_square, white_all_effect, long_check)

        # マスごとの利きの個数を計算
        count = BlackEffectCountLayer()(all_effects)

        return all_effects, count, all_check, available_square
