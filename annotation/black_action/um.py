#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt

from .major import BlackPromotedMajorMove

__author__ = 'Yasuhiro'
__date__ = '2018/2/24'


class BlackUmMoveLayer(snt.AbstractModule):
    def __init__(self, name='black_um_move'):
        super().__init__(name=name)

    def _build(self, board, um_effect):
        return BlackPromotedMajorMove()(board, um_effect)
