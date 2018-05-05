#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import sonnet as snt

from annotation.direction import Direction, get_eight_directions
from annotation.piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/2/09'


class CombineLayer(snt.AbstractModule):
    def __init__(self, name='combine'):
        super().__init__(name=name)

    def _build(self, effects):
        """
        長さ8の配列を一つにまとめる

        :param effects:
        :return:
        """
        tmp1 = [tf.logical_or(e1, e2)
                for e1, e2 in zip(effects[0::2], effects[1::2])]
        tmp2 = [tf.logical_or(e1, e2)
                for e1, e2 in zip(tmp1[0::2], tmp1[1::2])]
        effect = tf.logical_or(tmp2[0], tmp2[1])
        return effect


class CombineMovableMaskLayer(snt.AbstractModule):
    def __init__(self, name='combine_movable_mask'):
        super().__init__(name=name)

    def _build(self, effects):
        """
        長さ8の配列を一つにまとめる
        王手されている場合に動ける場所のマスクを計算する
        王手の場合はそのマスにしか動けないので、andの計算で絞り込む
        2方向から王手されている可能性もあるので、その場合は全てFalseになる

        :param effects:
        :return:
        """
        tmp1 = [tf.logical_and(e1, e2)
                for e1, e2 in zip(effects[0::2], effects[1::2])]
        tmp2 = [tf.logical_and(e1, e2)
                for e1, e2 in zip(tmp1[0::2], tmp1[1::2])]
        effect = tf.logical_and(tmp2[0], tmp2[1])
        return effect


class BlackCombineEffectLayer(snt.AbstractModule):
    def __init__(self, name='black_combine_effect'):
        super().__init__(name=name)

    def _build(self, effects):
        """
        駒の種類ごとに分けて計算されている利きをまとめる

        :param effects:
        :return:
        """
        # TO, NY, NK, NGはKIと同じ動きなので、KIに含まれている
        pieces0 = [Piece.BLACK_OU, Piece.BLACK_UM, Piece.BLACK_RY]
        pieces1 = [Piece.BLACK_GI, Piece.BLACK_KA, Piece.BLACK_KI] + pieces0
        pieces2 = [Piece.BLACK_HI, Piece.BLACK_KI] + pieces0
        pieces3 = [Piece.BLACK_GI, Piece.BLACK_KA] + pieces0

        outputs = dict((
            get_effects(
                direction=Direction.RIGHT_UP, pieces=pieces1, effects=effects
            ),
            get_effects(
                direction=Direction.RIGHT, pieces=pieces2, effects=effects
            ),
            get_effects(
                direction=Direction.RIGHT_DOWN, pieces=pieces3, effects=effects
            ),
            get_effects(
                direction=Direction.UP,
                pieces=[Piece.BLACK_FU, Piece.BLACK_KY, Piece.BLACK_GI,
                        Piece.BLACK_HI, Piece.BLACK_KI] + pieces0,
                effects=effects
            ),
            get_effects(
                direction=Direction.DOWN,
                pieces=[Piece.BLACK_HI, Piece.BLACK_KI] + pieces0,
                effects=effects
            ),
            get_effects(
                direction=Direction.LEFT_UP, pieces=pieces1, effects=effects
            ),
            get_effects(
                direction=Direction.LEFT, pieces=pieces2, effects=effects
            ),
            get_effects(
                direction=Direction.LEFT_DOWN, pieces=pieces3, effects=effects
            ),
            get_effects(
                direction=Direction.RIGHT_UP_UP, pieces=[Piece.BLACK_KE],
                effects=effects
            ),
            get_effects(
                direction=Direction.LEFT_UP_UP, pieces=[Piece.BLACK_KE],
                effects=effects
            )
        ))

        return outputs


def combine_or(*args):
    if len(args) == 1:
        return args[0]
    elif len(args) == 2:
        return tf.logical_or(*args)

    # 2個ずつまとめる
    outputs = [tf.logical_or(v1, v2) for v1, v2 in zip(args[0::2], args[1::2])]
    if len(args) % 2 == 1:
        # 奇数なので、最後の要素が余っている
        # 次に回す
        outputs.append(args[-1])

    return combine_or(*outputs)


def get_effects(direction, pieces, effects):
    """
    駒の種類ごとに計算した利きを方向ごとにまとめる

    :param direction:
    :param pieces:
    :param effects:
    :return:
    """
    short_effects = []
    long_effects = []
    for piece in pieces:
        effect = effects[piece][direction]
        if isinstance(effect, list):
            long_effects.append(effect)
        else:
            short_effects.append(effect)

    if direction not in get_eight_directions():
        # 桂馬の動き
        # dictを作りやすくするために方向と一緒に返す
        return direction, combine_or(*short_effects)

    long_merged = [tf.logical_or(v1, v2)
                   for v1, v2 in zip(long_effects[0], long_effects[1])]
    if len(long_effects):
        long_merged = [tf.logical_or(v1, v2)
                       for v1, v2 in zip(long_effects[2], long_merged)]

    short_effects.append(long_merged[0])
    short_merged = combine_or(*short_effects)
    long_merged[0] = short_merged

    # dictを作りやすくするために方向と一緒に返す
    return direction, long_merged
