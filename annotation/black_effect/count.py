#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sonnet as snt
import tensorflow as tf

__author__ = 'Yasuhiro'
__date__ = '2018/2/22'


class BlackEffectCountLayer(snt.AbstractModule):
    def __init__(self, name='black_effect_count'):
        super().__init__(name=name)

    def _build(self, all_effects):
        """
        手番側の利きがマスごとにいくつあるかを計算する

        :param all_effects:
        :return:
        """
        flat_effects = FlatEffectLayer()(all_effects)

        converted = [tf.to_int32(effect) for effect in flat_effects]
        count = tf.add_n(converted)

        return count


class FlatEffectLayer(snt.AbstractModule):
    def __init__(self, name='flat_effect'):
        super().__init__(name=name)

    def _build(self, all_effects):
        """
        駒ごと、方向ごとに入れ子になった利きを平らなlistに変換する

        :param all_effects:
        :return:
        """
        effect_list = []
        for effects in all_effects.values():
            for effect in effects.values():
                if isinstance(effect, list):
                    effect_list.extend(effect)
                else:
                    effect_list.append(effect)

        return effect_list
