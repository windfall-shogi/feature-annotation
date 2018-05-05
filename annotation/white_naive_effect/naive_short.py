#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt

from ..short_board.white_piece import select_white_short_pieces
from ..naive_effect import ShortEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/2/17'


class WhiteNaiveShortEffectLayer(snt.AbstractModule):
    def __init__(self, direction, data_format, use_cudnn=True,
                 name='white_naive_short_effect'):
        super().__init__(name=name)
        self.direction = direction
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, board):
        """
        手番側の王の利きを求めるためにピンや自殺手を考慮しない非手番側の利きを求める
        手番側の王の利きを求めるレイヤー内で1度だけ使用される

        :param board:
        :return:
        """
        # 方向directionへ動ける駒が1、他は全て0
        selected_pieces = select_white_short_pieces(board=board,
                                                    direction=self.direction)
        effect = ShortEffectLayer(
            direction=self.direction, data_format=self.data_format,
            use_cudnn=self.use_cudnn,
            name='white_naive_short_effect_{}'.format(self.direction.name)
        )(selected_pieces)

        return effect
