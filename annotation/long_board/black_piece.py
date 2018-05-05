#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from .axis import make_one_hot
from ..direction import (Direction, PinDirection, get_cross_directions,
                         get_diagonal_directions)
from ..piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/2/16'


def select_black_ou(board, data_format):
    """
    手番側の王とその他の駒をone hotで取り出す
    王はchannel1、その他の駒はchannel0

    王から利きを伸ばすことで、ピンされているかを判定に使う
    また、王手の検出にも使う
    王手されている場合は、利きのあるマスが王手を防ぐために他の駒を移動させる目的地になる

    :param board:
    :param data_format:
    :return:
    """
    table = np.zeros(Piece.SIZE, dtype=np.int32)
    table[Piece.BLACK_OU] = 1
    table[Piece.EMPTY] = -1

    converted = tf.gather(table, board)
    one_hot = make_one_hot(index_board=converted, data_format=data_format)

    return one_hot


def select_black_cross_pieces(board, data_format, direction):
    """
    非手番側の駒がピンされているかを判定するためにナイーブな利きを求める

    :param board:
    :param data_format:
    :param direction:
    :return:
    """
    if direction != Direction.UP:
        # 上方向は一回きりなので、残りの方向は保存しているか確認する
        collection = tf.get_collection_ref('black_cross_pieces')
        if len(collection):
            return collection[0]

    table = np.zeros(Piece.SIZE, dtype=np.int32)
    table[Piece.BLACK_HI] = 1
    table[Piece.BLACK_RY] = 1
    if direction == Direction.UP:
        table[Piece.BLACK_KY] = 1
    table[Piece.EMPTY] = -1

    converted = tf.gather(table, board)
    one_hot = make_one_hot(index_board=converted, data_format=data_format)

    if direction != Direction.UP:
        # 上方向は一回きりなので、残りの方向の場合は保存する
        tf.add_to_collection('black_cross_pieces', one_hot)

    return one_hot


# noinspection PyUnusedLocal
def select_black_diagonal_pieces(board, data_format, direction):
    """
    非手番側の駒がピンされているかを判定するためにナイーブな利きを求める

    :param board:
    :param data_format:
    :param direction:
    :return:
    """
    name = 'black_diagonal_pieces'
    collection = tf.get_collection_ref(name)
    if len(collection):
        return collection[0]

    table = np.zeros(Piece.SIZE, dtype=np.int32)
    table[Piece.BLACK_KA] = 1
    table[Piece.BLACK_UM] = 1
    table[Piece.EMPTY] = -1

    converted = tf.gather(table, board)
    one_hot = make_one_hot(index_board=converted, data_format=data_format)

    tf.add_to_collection(name, one_hot)

    return one_hot


def select_black_long_pieces(board, data_format, direction):
    if direction in get_cross_directions():
        return select_black_cross_pieces(
            board=board, data_format=data_format, direction=direction
        )
    elif direction in get_diagonal_directions():
        return select_black_diagonal_pieces(
            board=board, data_format=data_format, direction=direction
        )
    else:
        raise ValueError(direction)


def select_black_ky(board, data_format, direction):
    """
    ピンを考慮してKYを選び出す
    動けるKYはchanel1、動けないKYとその他の駒はchannel0にフラグを立てる

    :param board:
    :param data_format:
    :param direction:
    :return:
    """
    if direction == Direction.UP:
        table = make_ky_table(direction=direction)
        converted = tf.gather(table, board)

        one_hot = make_one_hot(index_board=converted, data_format=data_format)
        return one_hot
    else:
        raise ValueError(direction)


def select_black_ka(board, data_format, direction):
    """
    ピンを考慮してKAを選び出す
    動けるKAはchannel1、動けないKAとその他の駒はchannel0にフラグを立てる

    :param board:
    :param data_format:
    :param direction:
    :return:
    """
    return select_black_major_piece(board=board, data_format=data_format,
                                    direction=direction, piece=Piece.BLACK_KA)


def select_black_hi(board, data_format, direction):
    """
    ピンを考慮してHIを選び出す
    動けるHIはchannel1、動けないHIとその他の駒はchannel0にフラグを立てる

    :param board:
    :param data_format:
    :param direction:
    :return:
    """
    return select_black_major_piece(board=board, data_format=data_format,
                                    direction=direction, piece=Piece.BLACK_HI)


def select_black_um(board, data_format, direction):
    """
    ピンを考慮してKYを選び出す
    動けるUMはchannel1、動けないUMとその他の駒はchannel0にフラグを立てる

    :param board:
    :param data_format:
    :param direction:
    :return:
    """
    return select_black_major_piece(board=board, data_format=data_format,
                                    direction=direction, piece=Piece.BLACK_UM)


def select_black_ry(board, data_format, direction):
    """
    ピンを考慮してHIを選び出す
    動けるRYはchannel1、動けないRYとその他の駒はchannel0にフラグを立てる

    :param board:
    :param data_format:
    :param direction:
    :return:
    """
    return select_black_major_piece(board=board, data_format=data_format,
                                    direction=direction, piece=Piece.BLACK_RY)


def select_black_major_piece(board, data_format, direction, piece):
    if direction in get_directions(piece=piece):
        # ピンの方向ごとに計算をする
        # 4方向あるが、2方向でいい
        normalized_direction = PinDirection[direction.name]
        name = '{}_{}'.format(piece.name, normalized_direction.name)
        collection = tf.get_collection_ref(name)
        if len(collection):
            return collection[0]

        table = make_major_table(
            direction=direction, piece=piece,
            pin_directions=get_pin_directions(piece=piece)
        )
        converted = tf.gather(table, board)

        one_hot = make_one_hot(index_board=converted, data_format=data_format)

        tf.add_to_collection(name, one_hot)
        return one_hot
    else:
        raise ValueError(direction)


def get_directions(piece):
    if piece in (Piece.BLACK_KA, Piece.BLACK_UM):
        return get_diagonal_directions()
    elif piece in (Piece.BLACK_HI, Piece.BLACK_RY):
        return get_cross_directions()
    else:
        raise ValueError(piece)


def get_pin_directions(piece):
    if piece in (Piece.BLACK_KA, Piece.BLACK_UM):
        return PinDirection.DIAGONAL1, PinDirection.DIAGONAL2
    elif piece in (Piece.BLACK_HI, Piece.BLACK_RY):
        return PinDirection.VERTICAL, PinDirection.HORIZONTAL
    else:
        raise ValueError(piece)


def make_ky_table(direction):
    """
    ピンを考慮してKYの動きを計算する
    成りの計算のために駒の種類ごとに計算する

    :param direction:
    :return:
    """
    table = np.zeros(Piece.SIZE + 14 * PinDirection.SIZE, dtype=np.int32)
    if direction != Direction.UP:
        return table

    table[Piece.EMPTY] = -1

    table[Piece.BLACK_KY] = 1
    offset = Piece.SIZE + 14 * PinDirection.UP
    table[Piece.BLACK_KY + offset] = 1

    return table


def make_major_table(direction, piece, pin_directions):
    table = np.zeros(Piece.SIZE + 14 * PinDirection.SIZE, dtype=np.int32)

    normalized_direction = PinDirection[direction.name]
    name = 'black_{}_{}'.format(piece.name, normalized_direction.name)
    collection = tf.get_collection_ref(name)
    if len(collection):
        return collection[0]

    table[Piece.EMPTY] = -1
    table[piece] = 1
    # ピンの方向に対応する移動方向ならフラグを立てる
    offset = Piece.SIZE + piece - Piece.BLACK_FU
    for pin_direction in pin_directions:
        index = offset + 14 * pin_direction
        table[index] = pin_direction == normalized_direction

    tf.add_to_collection(name, table)

    return table
