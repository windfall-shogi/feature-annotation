#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from ..piece import Piece

__author__ = 'Yasuhiro'
__date__ = '2018/2/15'


def select_empty_board(board):
    """
    手番側が駒を打つために空いているマスを探す

    :param board:
    :return:
    """
    name = 'empty_board'
    collection = tf.get_collection_ref(name)
    if len(collection):
        return collection[0]

    selected = tf.equal(board, Piece.EMPTY)

    tf.add_to_collection(name, selected)

    return selected
