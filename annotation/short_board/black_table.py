#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from ..direction import Direction, PinDirection
from ..piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/3/09'


def make_black_direction_table(direction):
    if direction in (Direction.RIGHT_DOWN_DOWN, Direction.LEFT_DOWN_DOWN):
        raise ValueError(direction)

    name = 'black_naive_direction_table_{}'.format(direction.name)
    collection = tf.get_collection_ref(name)
    if len(collection):
        return collection[0]

    base = make_base_table()

    table = np.zeros(Piece.SIZE + 4 * 14)
    table[Piece.BLACK_FU:Piece.WHITE_FU] = base[direction]

    if direction not in (Direction.RIGHT_UP_UP, Direction.LEFT_UP_UP):
        offset = Piece.SIZE + PinDirection[direction.name] * 14
        table[offset:offset + 14] = base[direction]

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
        [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        # right
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        # right down
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        # up
        [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        # down
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        # left up
        [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        # left
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        # left down
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        # right up up
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # left up up
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # right down down
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # left down down
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.float32)

    return table
