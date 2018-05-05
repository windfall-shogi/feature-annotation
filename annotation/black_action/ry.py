#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt

from .major import BlackPromotedMajorMove

__author__ = 'Yasuhiro'
__date__ = '2018/2/24'


class BlackRyMoveLayer(snt.AbstractModule):
    def __init__(self, name='black_ry_move'):
        super().__init__(name=name)

    def _build(self, board, ry_effect):
        return BlackPromotedMajorMove()(board, ry_effect)
