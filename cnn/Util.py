#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as numpy
import tensorflow as tf

class Util(object):
  @staticmethod
  def tf_weight_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=\
      tf.contrib.layers.variance_scaling_initializer())

  @staticmethod
  def tf_bias_variable(name, shape):
    return tf.Variable(tf.constant(0.01, shape=shape), name=name)

  @staticmethod
  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  @staticmethod
  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  @staticmethod
  def decaying_learning_rate(min_lr, max_lr, lr_decay_speed, train_step):
    return min_lr + (max_lr - min_lr) * math.exp(-train_step / float(lr_decay_speed))
