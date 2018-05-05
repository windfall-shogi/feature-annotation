#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sonnet as snt

from ..long_board import white_piece as long_piece
from ..short_board import white_piece as short_piece
from ..direction import Direction, get_eight_directions
from ..naive_effect import LongEffectAllRangeLayer, ShortEffectLayer

__author__ = 'Yasuhiro'
__date__ = '2018/3/15'


class WhitePseudoOuEffect(snt.AbstractModule):
    def __init__(self, data_format, use_cudnn, name='white_ou_pseudo_effect'):
        super().__init__(name=name)
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, board):
        """
        非手番側の王から仮想的な利きを伸ばす
        利きがどの駒に当たっているかで王手されているかを判定する

        :param board:
        :return:
        """
        outputs = {}

        # 8方向について仮想的な利きを計算する
        ou_long_piece = long_piece.select_white_ou(
            board=board, data_format=self.data_format
        )
        for direction in get_eight_directions():
            effect = LongEffectAllRangeLayer(
                direction=direction, data_format=self.data_format,
                use_cudnn=self.use_cudnn,
                name='white_pseudo_ou_long_{}'.format(direction.name)
            )(ou_long_piece)
            outputs[direction] = effect

        # 桂馬の動き
        ou_short_piece = short_piece.select_white_ou(board=board)
        for direction in (Direction.RIGHT_DOWN_DOWN, Direction.LEFT_DOWN_DOWN):
            effect = ShortEffectLayer(
                direction=direction, data_format=self.data_format,
                use_cudnn=self.use_cudnn,
                name='white_pseudo_ou_short_{}'.format(direction.name)
            )(ou_short_piece)
            outputs[direction] = effect

        return outputs
