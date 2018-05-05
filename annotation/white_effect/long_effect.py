#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sonnet as snt

from ..direction import get_eight_directions
from ..long_board.white_piece import select_white_long_pieces
from ..naive_effect import LongEffectAllRangeLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/20'


class WhiteLongEffectLayer(snt.AbstractModule):
    def __init__(self, data_format, use_cudnn=True, name='white_long_effect'):
        super().__init__(name=name)
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, pinned_board):
        """
        全ての方向の利きをまとめる

        :param pinned_board:
        :return:
        """
        effect_list = [
            WhiteLongDirectedEffectLayer(
                direction=direction, data_format=self.data_format,
                use_cudnn=self.use_cudnn
            )(pinned_board) for direction in get_eight_directions()
        ]
        return effect_list


class WhiteLongDirectedEffectLayer(snt.AbstractModule):
    def __init__(self, direction, data_format, use_cudnn=True,
                 name='white_long_directed_effect'):
        super().__init__(name=name)
        self.direction = direction
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, pinned_board):
        """
        指定された方向の長い利きを求める

        :param pinned_board:
        :return:
        """
        selected = select_white_long_pieces(
            board=pinned_board, data_format=self.data_format,
            direction=self.direction, naive=False
        )
        effect = LongEffectAllRangeLayer(
            direction=self.direction, data_format=self.data_format,
            use_cudnn=self.use_cudnn
        )(selected)
        return effect
