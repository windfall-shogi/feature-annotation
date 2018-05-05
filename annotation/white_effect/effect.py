#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sonnet as snt

from .count import WhiteEffectCountLayer
from .long_effect import WhiteLongEffectLayer
from .ou import WhiteOuEffectLayer
from .pseudo_ou_effect import WhitePseudoOuEffect
from .short_effect import WhiteShortEffectLayer
from ..black_naive_effect import BlackNaiveAllEffect
from ..pin import WhitePinLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/20'


class WhiteEffectLayer(snt.AbstractModule):
    def __init__(self, data_format, use_cudnn=True, name='white_effect'):
        super().__init__(name=name)
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, board):
        """
        非手番側の有効な利きを求める

        :param board:
        :return:
        """
        # 非手番側のナイーブな利きを求める
        black_all_effect, black_long_effect = BlackNaiveAllEffect(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(board)

        pseudo_effect = WhitePseudoOuEffect(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(board)

        # ピンされているかを判定
        pinned_board = WhitePinLayer(
            data_format=self.data_format
        )(board, pseudo_effect, black_long_effect)

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
        )(board, black_all_effect)

        # マスごとの利きの個数を計算
        count = WhiteEffectCountLayer()(
            white_short_effect, white_long_effect, white_ou_effect
        )

        return count
