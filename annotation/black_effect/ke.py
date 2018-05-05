#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from ..direction import Direction
from ..naive_effect import ShortEffectLayer
from ..short_board.black_piece import select_black_ke

__author__ = 'Yasuhiro'
__date__ = '2018/2/19'


class BlackKeEffectLayer(snt.AbstractModule):
    def __init__(self, data_format, use_cudnn=True, name='black_ke_effect'):
        super().__init__(name=name)
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, pinned_board, available_square):
        outputs = {
            direction: self._make_effect(
                pinned_board=pinned_board, available_square=available_square,
                direction=direction
            ) for direction in (Direction.RIGHT_UP_UP, Direction.LEFT_UP_UP)
        }

        return outputs

    def _make_effect(self, pinned_board, available_square, direction):
        selected = select_black_ke(board=pinned_board, direction=direction)
        effect = ShortEffectLayer(
            direction=direction, data_format=self.data_format,
            use_cudnn=self.use_cudnn
        )(selected)
        return tf.logical_and(effect, available_square)
