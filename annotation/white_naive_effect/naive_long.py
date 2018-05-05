#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt

from ..long_board.white_piece import select_white_long_pieces
from ..naive_effect import LongEffectAllRangeLayer

__author__ = 'Yasuhiro'
__date__ = '2018/2/17'


class WhiteNaiveLongEffectLayer(snt.AbstractModule):
    def __init__(self, direction, data_format, use_cudnn=True,
                 name='white_naive_long_effect'):
        super().__init__(name=name)
        self.direction = direction
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, board):
        """
        手番側の王の利きを求めるためにピンや自殺手を考慮しない利きを求める
        手番側の王の利きを求める以外にも手番側の駒がピンされているかを判定するのにも使う

        :param board:
        :return:
        """
        selected_pieces = select_white_long_pieces(
            board=board, data_format=self.data_format, direction=self.direction,
            naive=True
        )

        name = 'white_naive_long_effect_all_range_{}'.format(
            self.direction.name
        )
        effect = LongEffectAllRangeLayer(
            direction=self.direction, data_format=self.data_format,
            use_cudnn=self.use_cudnn, name=name
        )(selected_pieces)

        return effect
