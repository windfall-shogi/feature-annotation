#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain

import sonnet as snt
import tensorflow as tf

from ..direction import get_eight_directions, Direction
from .combine import CombineMovableMaskLayer

__author__ = 'Yasuhiro'
__date__ = '2018/2/17'


class CheckAvailableSquareLayer(snt.AbstractModule):
    def __init__(self, name='available_square'):
        super().__init__(name=name)

    def _build(self, pseudo_ou_effect, check):
        """
        王手の場合には王手を防ぐ行動のみが有効になるので、そのためのマスクを計算する

        :param pseudo_ou_effect: 短い利きと長い利きで分かれている namedtuple
        :param check:
        :return:
        """
        # 王手がない場合は全ての動きが有効なので、その時の候補のマスク
        # サイズを取得するだけなので、方向は関係ない
        full = tf.ones_like(pseudo_ou_effect.long[Direction.UP], dtype=tf.bool)

        # 8方向に関しては長い利きを調べるだけで十分
        mask_list = [
            get_available_mask(
                pseudo_ou_effect=pseudo_ou_effect.long[direction],
                check=check[direction], full_mask=full
            ) for direction in get_eight_directions()
        ]
        # 桂馬の動きの場合
        mask_list.extend([
            get_available_mask(
                pseudo_ou_effect=pseudo_ou_effect.short[direction],
                check=check[direction], full_mask=full
            ) for direction in (Direction.RIGHT_UP_UP, Direction.LEFT_UP_UP)
        ])

        # 全ての方向で有効な利きがあるマスを選び出す
        mask = CombineMovableMaskLayer()(mask_list)
        return mask


def get_available_mask(pseudo_ou_effect, check, full_mask):
    # checkの方向はpseudo_ou_effectに合わせてある
    with_check = tf.logical_and(check, pseudo_ou_effect)
    without_check = tf.logical_and(tf.logical_not(check), full_mask)
    # 少なくとも片方は全部Falseになっているので、orで合わせる
    flag = tf.logical_or(with_check, without_check)

    return flag
