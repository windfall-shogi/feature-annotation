#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from .black_table import make_black_direction_table
from ..direction import (PinDirection, Direction, get_cross_directions,
                         get_diagonal_directions)
from ..piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/2/16'


def select_black_short_pieces(board, direction):
    """
    利きの短い駒を選び出す
    利きの長い駒は含まない

    非手番側の王が移動できる位置を求めるために手番側の利きがあるマスを求めるのに使う

    select_black_pieceと異なり指定された方向に移動できる短いの駒を全て選び出す

    :param board:
    :param direction:
    :return:
    """
    table = make_black_direction_table(direction=direction)
    converted = tf.gather(table, board)

    return converted


def select_black_ou(board):
    """
    手番側の有効な王の利きを求めるために使う

    :param board:
    :return:
    """
    # 桂馬で王手されているかを調べるために、擬似的に王から桂馬の効きを計算する
    # 王の通常の動きの計算もある
    # 2回使うので、collectionに登録する
    name = 'black_short_ou'
    collection = tf.get_collection_ref(name)
    if len(collection) == 0:
        selected = tf.to_float(tf.equal(board, Piece.BLACK_OU))
        tf.add_to_collection(name, selected)
    else:
        selected = collection[0]
    return selected


def make_mask(piece):
    mask = np.zeros(Piece.SIZE + 14 * PinDirection.SIZE)
    # 利用する場所だけを設定
    mask[piece] = 1
    mask[Piece.SIZE + piece::14] = 1

    return mask


def make_mask_ki():
    mask = np.zeros(Piece.SIZE + 14 * PinDirection.SIZE)
    for piece in (Piece.BLACK_KI, Piece.BLACK_TO, Piece.BLACK_NY,
                  Piece.BLACK_NK, Piece.BLACK_NG):
        mask[piece] = 1
        mask[Piece.SIZE + piece::14] = 1

    return mask


def select_black_piece(board, piece, direction):
    """
    成りの処理があるので、駒ごとに利きの判定を行う

    :param board:
    :param piece:
    :param direction:
    :return:
    """
    if piece == Piece.BLACK_KI:
        mask = make_mask_ki()
    else:
        mask = make_mask(piece=piece)

    table = mask * make_black_direction_table(direction=direction)
    selected = tf.gather(table, board)

    return selected


def select_black_fu(board, direction):
    """
    動けるFUが1、それ以外は0

    :param board:
    :param direction:
    :return:
    """
    if direction != Direction.UP:
        raise ValueError(direction)

    return select_black_piece(board=board, piece=Piece.BLACK_FU,
                              direction=direction)


def select_black_ke(board, direction):
    """
    動けるKEが1、それ以外は0

    :param board:
    :param direction:
    :return:
    """
    if direction not in (Direction.RIGHT_UP_UP, Direction.LEFT_UP_UP):
        raise ValueError(direction)

    # どちらの方向でも帰ってくる値は同じ
    name = 'black_ke_piece'
    collection = tf.get_collection_ref(name)
    if len(collection):
        return collection[0]

    ke = select_black_piece(board=board, piece=Piece.BLACK_KE,
                            direction=direction)
    tf.add_to_collection(name, ke)

    return ke


def select_black_gi(board, direction):
    """
    動けるGIが1、それ以外は0

    :param board:
    :param direction:
    :return:
    """
    if direction not in (Direction.RIGHT_UP, Direction.RIGHT_DOWN,
                         Direction.UP, Direction.LEFT_UP, Direction.LEFT_DOWN):
        raise ValueError(direction)

    # ピンの可能性があるので、方向ごとに計算し直す
    # ただし、右上と左下、右下と左上は同じ
    normalized_direction = PinDirection[direction.name]
    diagonals = PinDirection.DIAGONAL1, PinDirection.DIAGONAL2
    if normalized_direction in diagonals:
        name = 'black_gi_piece_{}'.format(normalized_direction.name)
        collection = tf.get_collection_ref(name)
        if len(collection):
            return collection[0]

    gi = select_black_piece(board=board, piece=Piece.BLACK_GI,
                            direction=direction)

    if normalized_direction in diagonals:
        # noinspection PyUnboundLocalVariable
        tf.add_to_collection(name, gi)

    return gi


def select_black_ki(board, direction):
    """
    動けるKI,TO,NY,NK,NGが1、それ以外は0

    :param board:
    :param direction:
    :return:
    """
    if direction not in (Direction.RIGHT_UP, Direction.RIGHT,
                         Direction.UP, Direction.DOWN,
                         Direction.LEFT_UP, Direction.LEFT):
        raise ValueError(direction)

    # ピンの可能性があるので、方向ごとに計算し直す
    # ただし、上と下、右と左は同じ値
    normalized_direction = PinDirection[direction.name]
    crosses = PinDirection.HORIZONTAL, PinDirection.VERTICAL
    if normalized_direction in crosses:
        name = 'black_ki_piece_{}'.format(normalized_direction.name)
        collection = tf.get_collection_ref(name)

        if len(collection):
            return collection[0]

    ki = select_black_piece(board=board, piece=Piece.BLACK_KI,
                            direction=direction)

    if normalized_direction in crosses:
        # noinspection PyUnboundLocalVariable
        tf.add_to_collection(name, ki)

    return ki


def select_black_um(board, direction):
    """
    UMの上下左右の短い利きのみについてピンを考慮して返す
    長い利きについては、long_boardの方の関数を利用する

    :param board:
    :param direction:
    :return:
    """
    if direction not in get_cross_directions():
        raise ValueError()

    normalized_direction = PinDirection[direction.name]
    name = 'black_um_piece_{}'.format(normalized_direction.name)
    collection = tf.get_collection_ref(name)

    if len(collection):
        return collection[0]

    um = select_black_piece(board=board, piece=Piece.BLACK_UM,
                            direction=direction)

    tf.add_to_collection(name, um)

    return um


def select_black_ry(board, direction):
    """
    RYの斜め方向の短い利きのみについてピンを考慮して返す
    長い利きについては、long_boardの方の関数を利用する

    :param board:
    :param direction:
    :return:
    """
    if direction not in get_diagonal_directions():
        raise ValueError()

    normalized_direction = PinDirection[direction.name]
    name = 'black_ry_piece_{}'.format(normalized_direction.name)
    collection = tf.get_collection_ref(name)

    if len(collection):
        return collection[0]

    ry = select_black_piece(board=board, piece=Piece.BLACK_RY,
                            direction=direction)

    tf.add_to_collection(name, ry)

    return ry
