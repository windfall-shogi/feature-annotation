#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from ..piece import Piece
from .white_table import (make_white_direction_table,
                          make_white_direction_table_without_ou)

__author__ = 'Yasuhiro'
__date__ = '2018/2/16'


def select_white_short_pieces(board, direction):
    """
    利きの短い駒を選び出す
    利きの長い駒は含まない

    手番側の王が移動できる位置を求めるために非手番側の利きがあるマスを求めるのに使う

    :param board:
    :param direction:
    :return:
    """
    table = make_white_direction_table(direction=direction)
    converted = tf.gather(table, board)

    return converted


def select_white_ou(board):
    """
    非手番側の有効な王の利きを求めるために使う

    :param board:
    :return:
    """
    # 桂馬で王手されているかを調べるために、擬似的に王から桂馬の効きを計算する
    # 王の通常の動きの計算もある
    # 2回使うので、collectionに登録する
    name = 'white_short_ou'
    collection = tf.get_collection_ref(name)
    if len(collection) == 0:
        selected = tf.to_float(tf.equal(board, Piece.WHITE_OU))
        tf.add_to_collection(name, selected)
    else:
        selected = collection[0]
    return selected


def select_white_short_pieces_without_ou(board, direction):
    """
    非手番側の利きを求めるために指定の方向に動ける駒を選び出す
    OUは手番側の利きを考慮する必要があるので、ここでは含めない

    :param board:
    :param direction:
    :return:
    """
    table = make_white_direction_table_without_ou(direction=direction)
    converted = tf.gather(table, board)

    return converted
