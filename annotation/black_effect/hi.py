#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from ..direction import get_cross_directions
from ..long_board.black_piece import select_black_hi
from ..naive_effect import LongEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/2/20'


class BlackHiEffectLayer(snt.AbstractModule):
    def __init__(self, data_format, use_cudnn=True, name='black_hi_effect'):
        super().__init__(name=name)
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, pinned_board, available_square):
        outputs = {
            direction: self._make_effect(
                pinned_board=pinned_board, available_square=available_square,
                direction=direction
            ) for direction in get_cross_directions()
        }

        return outputs

    def _make_effect(self, pinned_board, available_square, direction):
        selected = select_black_hi(
            board=pinned_board, data_format=self.data_format,
            direction=direction
        )
        effect_list = [LongEffectLayer(
            direction=direction, kernel_size=kernel_size,
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(selected) for kernel_size in range(2, 10)]

        return [tf.logical_and(effect, available_square)
                for effect in effect_list]
