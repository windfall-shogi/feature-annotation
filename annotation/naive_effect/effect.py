#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import sonnet as snt

from .kernel import make_short_kernel, make_long_kernel
from .pad import pad
from .combine import CombineLayer

__author__ = 'Yasuhiro'
__date__ = '2018/2/03'


class ShortEffectLayer(snt.AbstractModule):
    def __init__(self, direction, data_format, use_cudnn=True,
                 name='short_effect'):
        super().__init__(name=name)
        self.direction = direction
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, board):
        kernel = make_short_kernel(direction=self.direction)
        raw_value = tf.nn.conv2d(
            input=board, filter=kernel, strides=[1, 1, 1, 1], padding='VALID',
            use_cudnn_on_gpu=self.use_cudnn, data_format=self.data_format
        )
        outputs = pad(
            # 演算誤差はないと思うので、そのままbool型に変換する
            flag=tf.cast(raw_value, tf.bool), direction=self.direction,
            data_format=self.data_format, kernel_size=2
        )

        return outputs


class LongEffectLayer(snt.AbstractModule):
    def __init__(self, direction, kernel_size, data_format, use_cudnn=True,
                 name='long_effect'):
        super().__init__(name=name)
        self.direction = direction
        self.kernel_size = kernel_size
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, inputs):
        kernel = make_long_kernel(direction=self.direction,
                                  size=self.kernel_size)
        raw_value = tf.nn.conv2d(
            input=inputs, filter=kernel, strides=[1, 1, 1, 1], padding='VALID',
            use_cudnn_on_gpu=self.use_cudnn, data_format=self.data_format
        )
        # 1以上なら利きがある、0以下なら利きがない
        flag = raw_value > 0.5

        outputs = pad(
            flag=flag, direction=self.direction,
            data_format=self.data_format, kernel_size=self.kernel_size
        )

        return outputs


class LongEffectAllRangeLayer(snt.AbstractModule):
    def __init__(self, direction, data_format, use_cudnn=True,
                 name='long_effect_all_range'):
        super().__init__(name=name)
        self.direction = direction
        self.data_format = data_format
        self.use_cudnn = use_cudnn

    def _build(self, inputs):
        outputs = [
            LongEffectLayer(
                direction=self.direction, kernel_size=size,
                data_format=self.data_format, use_cudnn=self.use_cudnn,
                name='long_effect_{}{}'.format(self.direction.name, size - 2)
            )(inputs)
            for size in range(2, 10)
        ]

        outputs = CombineLayer()(outputs)
        return outputs
