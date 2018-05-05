#!/usr/bin/env python
# -*- coding: utf-8 -*-

import enum

__author__ = 'Yasuhiro'
__date__ = '2018/1/28'


class Direction(enum.IntEnum):
    RIGHT_UP = 0
    RIGHT = 1
    RIGHT_DOWN = 2
    UP = 3
    DOWN = 4
    LEFT_UP = 5
    LEFT = 6
    LEFT_DOWN = 7
    # KEの動き
    RIGHT_UP_UP = 8
    LEFT_UP_UP = 9
    RIGHT_DOWN_DOWN = 10
    LEFT_DOWN_DOWN = 11


def get_eight_directions():
    return (Direction.RIGHT_UP, Direction.RIGHT, Direction.RIGHT_DOWN,
            Direction.UP, Direction.DOWN,
            Direction.LEFT_UP, Direction.LEFT, Direction.LEFT_DOWN)


def get_cross_directions():
    return Direction.RIGHT, Direction.UP, Direction.DOWN, Direction.LEFT


def get_diagonal_directions():
    return (Direction.RIGHT_UP, Direction.RIGHT_DOWN,
            Direction.LEFT_UP, Direction.LEFT_DOWN)


def get_opposite_direction(direction):
    """
    逆方向を求める
    順方向、逆方向と動いた時に元の位置に戻れる動きを逆方向と定める

    :param direction:
    :return:
    """
    if direction < 8:
        return Direction(7 - direction)
    else:
        # 桂馬の動き
        return Direction(19 - direction)


class PinDirection(enum.IntEnum):
    DIAGONAL1 = 0
    HORIZONTAL = 1
    DIAGONAL2 = 2
    VERTICAL = 3

    SIZE = 4

    RIGHT_UP = 0
    RIGHT = 1
    RIGHT_DOWN = 2
    UP = 3
    DOWN = 3
    LEFT_UP = 2
    LEFT = 1
    LEFT_DOWN = 0


