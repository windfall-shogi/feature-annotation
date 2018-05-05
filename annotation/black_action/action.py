#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sonnet as snt

from .merge import BlackMergeAllLayer

__author__ = 'Yasuhiro'
__date__ = '2018/2/22'


class BlackActionLayer(snt.AbstractModule):
    def __init__(self, data_format, name='black_action'):
        super().__init__(name=name)
        self.data_format = data_format

    def _build(self, board, black_hand, all_effects, available_square):
        """
        行動を規定の順序で並べる

        :param board:
        :param black_hand:
        :param all_effects:
        :param available_square:
        :return:
        """
        all_actions = BlackMergeAllLayer(
            data_format=self.data_format
        )(board, black_hand, all_effects, available_square)

        # 特徴量の学習でまた分割するので、リストを繋げることはしない
        return all_actions
