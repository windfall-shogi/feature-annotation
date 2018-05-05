#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from ..piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/2/15'


def select_black_all_board(board):
    """
    手番側の有効な利きを求める処理の前処理としてピンされている手番側の駒を探す
    そのために利用する

    :param board:
    :return:
    """
    table = np.zeros(Piece.SIZE, dtype=np.bool)
    table[:Piece.WHITE_FU] = True

    selected = tf.gather(table, board)
    return selected


def select_non_black_board(board):
    """
    手番側が駒を動かすときに移動可能な場所であるかを判定する
    空いているマスか非手番側の駒のマスならTrue

    :param board:
    :return:
    """
    name = 'non_black_board'
    collection = tf.get_collection_ref(name)
    if len(collection):
        return collection[0]

    selected = tf.greater_equal(board, Piece.WHITE_FU)

    tf.add_to_collection(name, selected)

    return selected


def select_black_fu_board(board):
    """
    手番側のFUを打つためにFUの場所を調べる

    :param board:
    :return:
    """
    selected = tf.equal(board, Piece.BLACK_FU)
    return selected
