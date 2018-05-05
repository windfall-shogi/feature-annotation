#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sonnet as snt

from .fu import BlackFuEffectLayer
from .ky import BlackKyEffectLayer
from .ke import BlackKeEffectLayer
from .gi import BlackGiEffectLayer
from .ka import BlackKaEffectLayer
from .hi import BlackHiEffectLayer
from .ki import BlackKiEffectLayer
from .ou import BlackOuEffectLayer
from .um import BlackUmEffectLayer
from .ry import BlackRyEffectLayer
from ..piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/2/21'


class BlackAllEffectLayer(snt.AbstractModule):
    def __init__(self, data_format, use_cudnn=True, name='black_all_effect'):
        super().__init__(name=name)
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, pinned_board, available_square, white_naive_all_effect,
               white_long_check):
        """
        手番側の全ての駒の利きを計算する

        :param pinned_board:
        :param available_square:
        :param white_naive_all_effect:
        :param white_long_check:
        :return:
        """
        pieces = (Piece.BLACK_FU, Piece.BLACK_KY, Piece.BLACK_KE,
                  Piece.BLACK_GI, Piece.BLACK_KA, Piece.BLACK_HI,
                  Piece.BLACK_KI, Piece.BLACK_UM, Piece.BLACK_RY)
        classes = (BlackFuEffectLayer, BlackKyEffectLayer, BlackKeEffectLayer,
                   BlackGiEffectLayer, BlackKaEffectLayer, BlackHiEffectLayer,
                   BlackKiEffectLayer, BlackUmEffectLayer, BlackRyEffectLayer)

        outputs = {
            p: c(
                data_format=self.data_format, use_cudnn=self.use_cudnn
            )(pinned_board, available_square)
            for p, c in zip(pieces, classes)
        }
        outputs[Piece.BLACK_OU] = BlackOuEffectLayer(
            data_format=self.data_format, use_cudnn=self.use_cudnn
        )(pinned_board, white_naive_all_effect, white_long_check)

        return outputs
