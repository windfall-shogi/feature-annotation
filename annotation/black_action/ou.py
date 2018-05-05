#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

from ..boolean_board.black import select_non_black_board

__author__ = 'Yasuhiro'
__date__ = '2018/2/24'


class BlackOuMoveLayer(snt.AbstractModule):
    def __init__(self, name='black_ou_move'):
        super().__init__(name=name)

    def _build(self, board, ou_effect):
        non_black_mask = select_non_black_board(board=board)
        non_promoting_effect = {
            direction: tf.logical_and(non_black_mask, effect)
            for direction, effect in ou_effect.items()
        }

        return non_promoting_effect
