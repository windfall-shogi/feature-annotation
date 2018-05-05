#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from ..direction import Direction
from ..naive_effect import ShortEffectLayer
from ..short_board.black_piece import select_black_fu

__author__ = 'Yasuhiro'
__date__ = '2018/2/19'


class BlackFuEffectLayer(snt.AbstractModule):
    def __init__(self, data_format, use_cudnn=True, name='black_fu_effect'):
        super().__init__(name=name)
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, pinned_board, available_square):
        selected = select_black_fu(board=pinned_board, direction=Direction.UP)
        effect = ShortEffectLayer(
            direction=Direction.UP, data_format=self.data_format,
            use_cudnn=self.use_cudnn
        )(selected)
        outputs = {Direction.UP: tf.logical_and(effect, available_square)}

        return outputs
