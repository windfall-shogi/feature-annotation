#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from ..direction import get_eight_directions
from ..naive_effect import ShortEffectLayer
from ..short_board.white_piece import select_white_ou

__author__ = 'Yasuhiro'
__date__ = '2018/3/17'


class WhiteOuEffectLayer(snt.AbstractModule):
    def __init__(self, data_format, use_cudnn=True, name='white_ou_effect'):
        super().__init__(name=name)
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, board, black_naive_all_effect):
        selected = select_white_ou(board=board)
        outputs = []

        flipped_black_effect = tf.logical_not(black_naive_all_effect)
        for direction in get_eight_directions():
            effect = ShortEffectLayer(
                direction=direction, data_format=self.data_format,
                use_cudnn=self.use_cudnn
            )(selected)

            # effectでTrueのマスがある（盤の端で移動できない可能性がある）
            # そのマスが相手の効きと重なっていない
            # ならば、利きは有効
            available_effect = tf.logical_and(effect, flipped_black_effect)
            outputs.append(available_effect)

        return outputs
