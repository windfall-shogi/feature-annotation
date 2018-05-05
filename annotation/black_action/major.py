#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from .promotion_mask import make_promotion_mask
from ..boolean_board.black import select_non_black_board

__author__ = 'Yasuhiro'
__date__ = '2018/2/24'


class BlackMajorMove(snt.AbstractModule):
    def __init__(self, data_format, name='black_major_move'):
        super().__init__(name=name)
        self.data_format = data_format

    def _build(self, board, major_effect):
        non_black_mask = select_non_black_board(board=board)
        non_promoting_effect = {
            direction: [tf.logical_and(non_black_mask, effect)
                        for effect in effects]
            for direction, effects in major_effect.items()
        }

        promoting_effect = {
            direction: [
                tf.logical_and(
                    make_promotion_mask(
                        step_size=i, data_format=self.data_format,
                        direction=direction
                    ),
                    effect
                ) for i, effect in enumerate(effects, start=1)
            ] for direction, effects in non_promoting_effect.items()
        }

        return non_promoting_effect, promoting_effect


class BlackPromotedMajorMove(snt.AbstractModule):
    def __init__(self, name='black_promoted_major_move'):
        super().__init__(name=name)

    def _build(self, board, major_effect):
        non_black_mask = select_non_black_board(board=board)

        def _update(obj):
            if isinstance(obj, list):
                return [tf.logical_and(non_black_mask, e) for e in obj]
            else:
                return tf.logical_and(non_black_mask, obj)

        non_promoting_effect = {
            direction: _update(effects)
            for direction, effects in major_effect.items()
        }

        return non_promoting_effect
