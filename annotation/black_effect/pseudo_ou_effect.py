#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
from itertools import chain

import sonnet as snt

from .ou_helper import get_long_effect, get_short_effect
from ..direction import Direction, get_eight_directions

__author__ = 'Yasuhiro'
__date__ = '2018/2/18'


PseudoEffect = namedtuple('PseudoEffect', ['short', 'long'])


class BlackPseudoOuEffect(snt.AbstractModule):
    def __init__(self, data_format, use_cudnn, name='black_ou_pseudo_effect'):
        super().__init__(name=name)
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, board):
        """
        手番側の王から仮想的な利きを伸ばす
        利きがどの駒に当たっているかで王手されているかを判定する
        また、王手の場合に、王手を防ぐ手として動く目的地になる

        :param board:
        :return:
        """
        # 8方向について仮想的な長い利きを計算する
        long_effects = {
            direction:
                get_long_effect(
                    board=board, direction=direction,
                    data_format=self.data_format, use_cudnn=self.use_cudnn
                ) for direction in get_eight_directions()

        }

        # 桂馬の動きと通常の動きを計算する
        short_effects = {
            direction:
                get_short_effect(
                    board=board, direction=direction,
                    data_format=self.data_format, use_cudnn=self.use_cudnn
                ) for direction in chain(get_eight_directions(),
                                         (Direction.RIGHT_UP_UP,
                                          Direction.LEFT_UP_UP))
        }

        effects = PseudoEffect(short=short_effects, long=long_effects)
        return effects
