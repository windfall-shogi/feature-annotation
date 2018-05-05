#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from ..piece import Piece
from ..short_board.white_table import make_base_table

__author__ = 'Yasuhiro'
__date__ = '2018/2/15'


def select_white_ka_um_board(board):
    """
    手番側の王から仮想的な利きを伸ばした時に利きの長い駒に当たるかを調べるために使う
    利きが当たる場合は、手番側の王は王手されている

    駒があるかどうかを判定したいので、bool型に変換する

    :param board:
    :return:
    """
    table = np.zeros(Piece.SIZE, dtype=np.bool)
    table[Piece.WHITE_KA] = True
    table[Piece.WHITE_UM] = True

    selected = tf.gather(table, board)
    return selected


def select_white_hi_ry_board(board):
    """
    手番側の王から仮想的な利きを伸ばした時に利きの長い駒に当たるかを調べるために使う
    利きが当たる場合は、手番側の王は王手されている

    駒があるかどうかを判定したいので、bool型に変換する

    :param board:
    :return:
    """
    table = np.zeros(Piece.SIZE, dtype=np.bool)
    table[Piece.WHITE_HI] = True
    table[Piece.WHITE_RY] = True

    selected = tf.gather(table, board)
    return selected


def select_white_ky_board(board):
    """
    手番側の王から仮想的な利きを伸ばした時に利きの長い駒に当たるかを調べるために使う
    利きが当たる場合は、手番側の王は王手されている

    駒があるかどうかを判定したいので、bool型に変換する

    :param board:
    :return:
    """
    table = np.zeros(Piece.SIZE, dtype=np.bool)
    table[Piece.WHITE_KY] = True

    selected = tf.gather(table, board)
    return selected


def select_white_direction_board(board, direction):
    """
    手番側の王から利きを伸ばした時にそれぞれの方向に動ける駒があるかを調べるために使う
    短い利きで手番側の王が王手されているかを判定に使う

    駒があるかどうかを判定したいので、bool型に変換する

    :param board:
    :param direction:
    :return:
    """
    table = np.zeros(Piece.SIZE, dtype=np.bool)
    direction_table = make_base_table()
    table[Piece.WHITE_FU:Piece.EMPTY] = direction_table[direction]

    selected = tf.gather(table, board)
    return selected


def select_white_all_board(board):
    """
    非手番側の有効な利きを求める処理の前処理としてピンされている非手番側の駒を探す
    そのために利用する

    :param board:
    :return:
    """
    table = np.zeros(Piece.SIZE, dtype=np.bool)
    table[Piece.WHITE_FU:Piece.SIZE] = True

    selected = tf.gather(table, board)
    return selected
