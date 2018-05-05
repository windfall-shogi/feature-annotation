#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain

import sonnet as snt

from ..direction import Direction, get_eight_directions
from ..naive_effect import ShortEffectLayer
from ..short_board.white_piece import select_white_short_pieces_without_ou

__author__ = 'Yasuhiro'
__date__ = '2018/3/17'


class WhiteShortEffectLayer(snt.AbstractModule):
    def __init__(self, data_format, use_cudnn=True, name='white_short_effect'):
        super().__init__(name=name)
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, pinned_board):
        """
        王以外の全ての方向の利きをまとめる

        :param pinned_board:
        :return:
        """
        effect_list = [
            WhiteShortDirectedEffectLayer(
                direction=direction, data_format=self.data_format,
                use_cudnn=self.use_cudnn
            )(pinned_board)
            for direction in chain(get_eight_directions(),
                                   [Direction.RIGHT_DOWN_DOWN,
                                    Direction.LEFT_DOWN_DOWN])
        ]
        return effect_list


class WhiteShortDirectedEffectLayer(snt.AbstractModule):
    def __init__(self, direction, data_format, use_cudnn=True,
                 name='white_short_directed_effect'):
        super().__init__(name=name)
        self.direction = direction
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, pinned_board):
        """
        指定された方向の王以外の短い利きを求める

        :param pinned_board:
        :return:
        """
        selected = select_white_short_pieces_without_ou(
            board=pinned_board, direction=self.direction
        )
        effect = ShortEffectLayer(
            direction=self.direction, data_format=self.data_format,
            use_cudnn=self.use_cudnn
        )(selected)
        return effect
