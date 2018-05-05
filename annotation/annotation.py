#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sonnet as snt

from .black_effect import BlackEffectLayer
from .black_action import BlackActionLayer
from .white_effect import WhiteEffectLayer
from .black_action.merge import merge

__author__ = 'Yasuhiro'
__date__ = '2018/3/22'


class AnnotationLayer(snt.AbstractModule):
    def __init__(self, data_format, use_cudnn, name='annotation'):
        super().__init__(name=name)
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, board, black_hand):
        (black_all_effects, black_count, black_check,
         available_square) = BlackEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(board)
        all_actions = BlackActionLayer(
            data_format=self.data_format
        )(board, black_hand, black_all_effects, available_square)

        white_count = WhiteEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(board)

        # 方向ごとの王手の判定をまとめる
        check_list = list(black_check.values())
        black_check = merge(*check_list)

        return all_actions, black_count, white_count, black_check
