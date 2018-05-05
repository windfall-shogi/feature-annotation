#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from .ou_helper import get_short_effect
from ..direction import get_eight_directions, get_opposite_direction

__author__ = 'Yasuhiro'
__date__ = '2018/2/20'


class BlackOuEffectLayer(snt.AbstractModule):
    def __init__(self, data_format, use_cudnn=True, name='black_ou_effect'):
        super().__init__(name=name)
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, board, white_naive_all_effect, white_long_check):
        outputs = {}
        for direction in get_eight_directions():
            effect = get_short_effect(
                board=board, direction=direction, data_format=self.data_format,
                use_cudnn=self.use_cudnn
            )

            opposite_direction = get_opposite_direction(direction=direction)
            # effectでTrueのマスがある（盤の端で移動できない可能性がある）
            # そのマスが相手の効きと重なっていない
            # long_checkがFalse
            # ならば、利きは有効
            overlap = tf.reduce_any(
                tf.logical_and(effect, white_naive_all_effect),
                axis=[1, 2, 3], keep_dims=True
            )
            long_check = white_long_check[opposite_direction]
            available_effect = tf.logical_and(
                effect,
                tf.logical_not(tf.logical_or(overlap, long_check))
            )

            outputs[direction] = available_effect
        return outputs

