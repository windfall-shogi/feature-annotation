#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from ..direction import Direction
from ..long_board.black_piece import select_black_ky
from ..naive_effect import LongEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/2/19'


class BlackKyEffectLayer(snt.AbstractModule):
    def __init__(self, data_format, use_cudnn=True, name='black_ky_effect'):
        super().__init__(name=name)
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, pinned_board, available_square):
        selected = select_black_ky(
            board=pinned_board, data_format=self.data_format,
            direction=Direction.UP
        )
        effect_list = [
            LongEffectLayer(
                direction=Direction.UP, kernel_size=kernel_size,
                data_format=self.data_format, use_cudnn=self.use_cudnn
            )(selected) for kernel_size in range(2, 10)
        ]

        outputs = {
            Direction.UP: [tf.logical_and(effect, available_square)
                           for effect in effect_list]
        }

        return outputs
