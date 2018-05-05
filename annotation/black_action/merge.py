#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict

import sonnet as snt
import tensorflow as tf

from .fu import BlackFuMoveLayer, BlackFuDropLayer
from .ky import BlackKyMoveLayer, BlackKyDropLayer
from .ke import BlackKeMoveLayer, BlackKeDropLayer
from .gi import BlackGiMoveLayer, BlackGiDropLayer
from .ka import BlackKaMoveLayer, BlackKaDropLayer
from .hi import BlackHiMoveLayer, BlackHiDropLayer
from .ki import BlackKiMoveLayer, BlackKiDropLayer
from .ou import BlackOuMoveLayer
from .um import BlackUmMoveLayer
from .ry import BlackRyMoveLayer
from ..piece import Piece
from ..direction import get_eight_directions, Direction

__author__ = 'Yasuhiro'
__date__ = '2018/2/25'


class BlackMergeMoveLayer(snt.AbstractModule):
    def __init__(self, data_format, name='black_merge_move'):
        super().__init__(name=name)
        self.data_format = data_format

    def _build(self, board, all_effects):
        fu_move = BlackFuMoveLayer(
            data_format=self.data_format
        )(board, all_effects[Piece.BLACK_FU])
        ky_move = BlackKyMoveLayer(
            data_format=self.data_format
        )(board, all_effects[Piece.BLACK_KY])
        ke_move = BlackKeMoveLayer(
            data_format=self.data_format
        )(board, all_effects[Piece.BLACK_KE])
        gi_move = BlackGiMoveLayer(
            data_format=self.data_format
        )(board, all_effects[Piece.BLACK_GI])
        ka_move = BlackKaMoveLayer(
            data_format=self.data_format
        )(board, all_effects[Piece.BLACK_KA])
        hi_move = BlackHiMoveLayer(
            data_format=self.data_format
        )(board, all_effects[Piece.BLACK_HI])
        ki_move = BlackKiMoveLayer()(board, all_effects[Piece.BLACK_KI])
        ou_move = BlackOuMoveLayer()(board, all_effects[Piece.BLACK_OU])
        um_move = BlackUmMoveLayer()(board, all_effects[Piece.BLACK_UM])
        ry_move = BlackRyMoveLayer()(board, all_effects[Piece.BLACK_RY])

        # 桂馬だけは特別なので、別処理にする
        move_list = [fu_move, ky_move, gi_move, ka_move, hi_move,
                     ki_move, ou_move, um_move, ry_move]
        actions = [(defaultdict(list), defaultdict(list)) for _ in range(8)]

        for move in move_list:
            if not isinstance(move, tuple):
                move = move,

            for move_type, move_data in enumerate(move):
                for direction, obj in move_data.items():
                    if not isinstance(obj, list):
                        obj = obj,

                    for distance, action in enumerate(obj):
                        actions[distance][move_type][direction].append(action)

        outputs = []
        for distance in range(8):
            non_promotion, promotion = actions[distance]
            # 成らない場合
            outputs.extend([merge(*non_promotion[direction])
                            for direction in get_eight_directions()])
            # 成る場合
            outputs.extend([merge(*promotion[direction])
                            for direction in get_eight_directions()])
        # 桂馬の動き
        outputs.append(ke_move[0][Direction.RIGHT_UP_UP])
        outputs.append(ke_move[0][Direction.LEFT_UP_UP])
        outputs.append(ke_move[1][Direction.RIGHT_UP_UP])
        outputs.append(ke_move[1][Direction.LEFT_UP_UP])

        return outputs


class BlackMergeDropLayer(snt.AbstractModule):
    def __init__(self, data_format, name='black_merge_drop'):
        super().__init__(name=name)
        self.data_format = data_format

    def _build(self, board, black_hand, available_square):
        fu_drop = BlackFuDropLayer(self.data_format)(board, black_hand,
                                                     available_square)
        ky_drop = BlackKyDropLayer(self.data_format)(board, black_hand,
                                                     available_square)
        ke_drop = BlackKeDropLayer(self.data_format)(board, black_hand,
                                                     available_square)
        gi_drop = BlackGiDropLayer(self.data_format)(board, black_hand,
                                                     available_square)
        ka_drop = BlackKaDropLayer(self.data_format)(board, black_hand,
                                                     available_square)
        hi_drop = BlackHiDropLayer(self.data_format)(board, black_hand,
                                                     available_square)
        ki_drop = BlackKiDropLayer(self.data_format)(board, black_hand,
                                                     available_square)

        outputs = [fu_drop, ky_drop, ke_drop, gi_drop,
                   ka_drop, hi_drop, ki_drop]
        return outputs


class BlackMergeAllLayer(snt.AbstractModule):
    def __init__(self, data_format, name='black_merge_all'):
        super().__init__(name=name)
        self.data_format = data_format

    def _build(self, board, black_hand, all_effects, available_square):
        moves = BlackMergeMoveLayer(
            data_format=self.data_format
        )(board, all_effects)
        drops = BlackMergeDropLayer(
            data_format=self.data_format
        )(board, black_hand, available_square)

        actions = moves + drops
        return actions


def merge(*args):
    n = len(args)
    if n == 1:
        return args[0]
    elif n == 2:
        return tf.logical_or(*args)

    tmp = [tf.logical_or(v1, v2) for v1, v2 in zip(args[0::2], args[1::2])]
    if n % 2:
        # 奇数の場合にペアに成らなかった要素を次回に繰り越す
        tmp.append(args[-1])

    return merge(*tmp)
