#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from ..naive_effect import ShortEffectLayer, LongEffectAllRangeLayer
from ..long_board import black_piece as long_piece
from ..short_board import black_piece as short_piece
from ..direction import Direction, get_eight_directions

__author__ = 'Yasuhiro'
__date__ = '2018/3/25'


def get_short_effect(board, direction, data_format, use_cudnn):
    # 桂馬の動きの場合は一度しか利用しないので、保存しない
    flag = direction in get_eight_directions()

    if flag:
        name = 'black_ou_short_move_{}'.format(direction.name)
        collection = tf.get_collection_ref(name)
        if len(collection):
            return collection[0]

    ou = get_short_ou(board=board)
    effect = ShortEffectLayer(
        direction=direction, data_format=data_format,
        use_cudnn=use_cudnn,
        name='black_ou_short_{}'.format(direction.name)
    )(ou)

    if flag:
        # noinspection PyUnboundLocalVariable
        tf.add_to_collection(name, effect)

    return effect


def get_short_ou(board):
    name = 'black_short_ou'
    collection = tf.get_collection_ref(name)
    if len(collection):
        return collection[0]

    ou_short_piece = short_piece.select_black_ou(board=board)

    tf.add_to_collection(name, ou_short_piece)

    return ou_short_piece


def get_long_effect(board, direction, data_format, use_cudnn):
    ou = get_long_ou(board=board, data_format=data_format)
    effect = LongEffectAllRangeLayer(
        direction=direction, data_format=data_format,
        use_cudnn=use_cudnn,
        name='black_pseudo_ou_short_{}'.format(direction.name)
    )(ou)

    return effect


def get_long_ou(board, data_format):
    name = 'black_long_ou'
    collection = tf.get_collection_ref(name)
    if len(collection):
        return collection[0]

    ou_long_piece = long_piece.select_black_ou(board=board,
                                               data_format=data_format)

    tf.add_to_collection(name, ou_long_piece)

    return ou_long_piece
