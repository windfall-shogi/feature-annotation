#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import enum

__author__ = 'Yasuhiro'
__date__ = '2018/1/27'


class Piece(enum.IntEnum):
    BLACK_FU = 0
    BLACK_KY = 1
    BLACK_KE = 2
    BLACK_GI = 3
    BLACK_KA = 4
    BLACK_HI = 5
    BLACK_KI = 6
    BLACK_OU = 7
    BLACK_TO = 8
    BLACK_NY = 9
    BLACK_NK = 10
    BLACK_NG = 11
    BLACK_UM = 12
    BLACK_RY = 13

    WHITE_FU = 14
    WHITE_KY = 15
    WHITE_KE = 16
    WHITE_GI = 17
    WHITE_KA = 18
    WHITE_HI = 19
    WHITE_KI = 20
    WHITE_OU = 21
    WHITE_TO = 22
    WHITE_NY = 23
    WHITE_NK = 24
    WHITE_NG = 25
    WHITE_UM = 26
    WHITE_RY = 27

    EMPTY = 28

    SIZE = 29
