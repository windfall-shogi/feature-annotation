#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from .axis import make_one_hot
from ..direction import (Direction, PinDirection, get_cross_directions,
                         get_diagonal_directions)
from ..piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/2/17'


def select_white_ou(board, data_format):
    """
    非手番側の王とその他の駒をone hotで取り出す
    王はchannel1、その他の駒はchannel0

    王から利きを伸ばすことで、ピンされているかを判定に使う

    :param board:
    :param data_format:
    :return:
    """
    table = np.zeros(Piece.SIZE, dtype=np.int32)
    table[Piece.WHITE_OU] = 1
    table[Piece.EMPTY] = -1

    converted = tf.gather(table, board)
    one_hot = make_one_hot(index_board=converted, data_format=data_format)

    return one_hot


def select_white_cross_pieces(board, data_format, direction, naive):
    """
    手番側の駒がピンされているかを判定するためにナイーブな利きを求める
    また、ピンを考慮した利きを求める

    :param board:
    :param data_format:
    :param direction:
    :param naive: naiveかどうかでboardの値が違うので、それを管理する
    :return:
    """
    if naive:
        fmt = 'white_naive_cross_pieces_{}'
    else:
        fmt = 'white_cross_pieces_{}'

    if direction in (Direction.RIGHT, Direction.LEFT):
        name = fmt.format(PinDirection.HORIZONTAL.name)
    else:
        name = fmt.format(direction.name)
    collection = tf.get_collection_ref(name)
    if len(collection):
        return collection[0]

    normalized_direction = PinDirection[direction.name]

    table = np.zeros(Piece.SIZE + 14 * PinDirection.SIZE, dtype=np.int32)
    table[Piece.WHITE_HI] = 1
    table[Piece.WHITE_RY] = 1
    for pin_direction in (PinDirection.HORIZONTAL, PinDirection.VERTICAL):
        offset = Piece.SIZE - Piece.WHITE_FU + 14 * pin_direction
        value = pin_direction == normalized_direction
        table[offset + Piece.WHITE_HI] = value
        table[offset + Piece.WHITE_RY] = value
    if direction == Direction.DOWN:
        table[Piece.WHITE_KY] = 1
        index = (Piece.SIZE + Piece.WHITE_KY - Piece.WHITE_FU +
                 14 * PinDirection.VERTICAL)
        table[index] = 1
    table[Piece.EMPTY] = -1

    converted = tf.gather(table, board)
    one_hot = make_one_hot(index_board=converted, data_format=data_format)

    tf.add_to_collection(name, one_hot)

    return one_hot


def select_white_diagonal_pieces(board, data_format, direction, naive):
    """
    手番側の駒がピンされているかを判定するためにナイーブな利きを求める
    また、ピンを考慮した利きを求める

    :param board:
    :param data_format:
    :param direction:
    :param naive: naiveかどうかでboardの値が違うので、それを管理する
    :return:
    """
    if naive:
        fmt = 'white_naive_diagonal_pieces_{}'
    else:
        fmt = 'white_diagonal_pieces_{}'

    normalized_direction = PinDirection[direction.name]
    name = fmt.format(normalized_direction.name)
    collection = tf.get_collection_ref(name)
    if len(collection):
        return collection[0]

    table = np.zeros(Piece.SIZE + 14 * PinDirection.SIZE, dtype=np.int32)
    table[Piece.WHITE_KA] = 1
    table[Piece.WHITE_UM] = 1
    for pin_direction in (PinDirection.DIAGONAL1, PinDirection.DIAGONAL2):
        offset = Piece.SIZE - Piece.WHITE_FU + 14 * pin_direction
        value = pin_direction == normalized_direction
        table[offset + Piece.WHITE_KA] = value
        table[offset + Piece.WHITE_UM] = value
    table[Piece.EMPTY] = -1

    converted = tf.gather(table, board)
    one_hot = make_one_hot(index_board=converted, data_format=data_format)

    tf.add_to_collection(name, one_hot)

    return one_hot


def select_white_long_pieces(board, data_format, direction, naive):
    """
    手番側の駒がピンされているかを判定するためにナイーブな利きを求める
    また、ピンを考慮した利きを求める

    :param board:
    :param data_format:
    :param direction:
    :param naive: naiveかどうかでboardの値が違うので、それを管理する
    :return:
    """
    if direction in get_cross_directions():
        return select_white_cross_pieces(
            board=board, data_format=data_format, direction=direction,
            naive=naive
        )
    elif direction in get_diagonal_directions():
        return select_white_diagonal_pieces(
            board=board, data_format=data_format, direction=direction,
            naive=naive
        )
    else:
        raise ValueError(direction)
