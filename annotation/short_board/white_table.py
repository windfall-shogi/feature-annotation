#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from ..direction import Direction, PinDirection
from ..piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/2/28'


def make_white_direction_table(direction):
    return _make_white_table(direction=direction, mask_ou=False)


def make_white_direction_table_without_ou(direction):
    return _make_white_table(direction=direction, mask_ou=True)


def _make_white_table(direction, mask_ou):
    if direction in (Direction.RIGHT_UP_UP, Direction.LEFT_UP_UP):
        raise ValueError(direction)

    name = 'white_naive_direction_table_{}'.format(direction.name)
    if mask_ou:
        name += 'without_ou'
    collection = tf.get_collection_ref(name)
    if len(collection):
        return collection[0]

    base = make_base_table()

    table = np.zeros(Piece.SIZE + PinDirection.SIZE * 14)
    table[Piece.WHITE_FU:Piece.EMPTY] = base[direction]

    if direction not in (Direction.RIGHT_DOWN_DOWN, Direction.LEFT_DOWN_DOWN):
        offset = Piece.SIZE + PinDirection[direction.name] * 14
        table[offset:offset + 14] = base[direction]

    if mask_ou:
        # OUを選ばない
        table[Piece.WHITE_OU] = 0
        table[Piece.SIZE + Piece.WHITE_OU - Piece.WHITE_FU::14] = 0

    table = tf.constant(table, dtype=tf.float32)

    tf.add_to_collection(name, table)

    return table


def make_base_table():
    """
    ピンを考えない状況での方向ごとの基本的な利きの有無を設定する
    長い利きは含まない

    :return:
    """
    table = np.array([
        # right up
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        # right
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        # right down
        [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        # up
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        # down
        [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        # left up
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        # left
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        # left down
        [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        # right up up
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # left up up
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # right down down
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # left down down
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.float32)

    return table
