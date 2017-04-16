#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import tensorflow as tf

from cnn.Util import Util

dir_path = os.path.dirname(os.path.realpath(__file__))

class ModelV1(object):
  def __init__(self, data, tensorboard_dir):
    super(ModelV1, self).__init__()

    self.tensorboard_dir = tensorboard_dir

    self.max_learning_rate = 0.00050
    self.min_learning_rate = 0.00005
    self.decay_speed = 20.0
    self.l2_regularizer_beta = 0.0125 # 0.0150?

    self.data = data
    self.sess = tf.Session()
    self.__build_network()

  def __write_summary(self, name_values, count):
    summaries = self.sess.run([self.summary_objs[t] for t in name_values], {
      self.summary_scalars[t] : name_values[t] for t in name_values
    })
    for s in summaries:
      self.writer.add_summary(s, count)
    print ', '.join(('%s : %f' % (t, name_values[t])) for t in name_values)

  def __build_network(self):
    self.params = dict()

    # ================ Input ================
    self.x = tf.placeholder(tf.float32, [None, 42 * 50])
    self.y = tf.placeholder(tf.float32, [None, 7])

    # ================ Dynamic params ================
    self.prob_keep_conv1 = tf.placeholder(tf.float32)
    self.prob_keep_conv2 = tf.placeholder(tf.float32)
    self.prob_keep_conv3 = tf.placeholder(tf.float32)
    self.prob_keep_fc1 = tf.placeholder(tf.float32)
    self.learning_rate = tf.placeholder(tf.float32)

    # ================ Params ================
    self.params['W_conv1'] = Util.tf_weight_variable('W_conv1', [7, 7, 1, 24])
    self.params['b_conv1'] = Util.tf_bias_variable('b_conv1', [24])

    self.params['W_conv2'] = Util.tf_weight_variable('W_conv2', [5, 5, 24, 24])
    self.params['b_conv2'] = Util.tf_bias_variable('b_conv2', [24])

    self.params['W_conv3'] = Util.tf_weight_variable('W_conv3', [3, 3, 24, 32])
    self.params['b_conv3'] = Util.tf_bias_variable('b_conv3', [32])

    self.params['W_fc1'] = Util.tf_weight_variable('W_fc1', [6 * 7 * 32, 256])
    self.params['b_fc1'] = Util.tf_bias_variable('b_fc1', [256])

    self.params['W_nn1'] = Util.tf_weight_variable('W_nn1', [256, 7])
    self.params['b_nn1'] = Util.tf_bias_variable('b_nn1', [7])

    # ================ Graph ================
    with tf.name_scope('network'):
      self.x_reshaped = tf.reshape(self.x, [-1, 42, 50, 1])
      self.h_conv1 = tf.nn.relu(Util.conv2d(self.x_reshaped, self.params['W_conv1']) + self.params['b_conv1'])
      self.h_conv1_drop = tf.nn.dropout(self.h_conv1, self.prob_keep_conv1)
      self.h_pool1 = Util.max_pool_2x2(self.h_conv1_drop)

      self.h_conv2 = tf.nn.relu(Util.conv2d(self.h_pool1, self.params['W_conv2']) + self.params['b_conv2'])
      self.h_conv2_drop = tf.nn.dropout(self.h_conv2, self.prob_keep_conv2)
      self.h_pool2 = Util.max_pool_2x2(self.h_conv2_drop)

      self.h_conv3 = tf.nn.relu(Util.conv2d(self.h_pool2, self.params['W_conv3']) + self.params['b_conv3'])
      self.h_conv3_drop = tf.nn.dropout(self.h_conv3, self.prob_keep_conv3)
      self.h_pool3 = Util.max_pool_2x2(self.h_conv3_drop)

      self.h_pool3_flat = tf.reshape(self.h_pool3, [-1, 6 * 7 * 32])
      self.h_fc1_b4_relu = tf.matmul(self.h_pool3_flat, self.params['W_fc1']) + self.params['b_fc1']
      self.h_fc1 = tf.nn.relu(self.h_fc1_b4_relu)
      self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.prob_keep_fc1)

      self.out_layer = tf.matmul(self.h_fc1_drop, self.params['W_nn1']) + self.params['b_nn1']

    with tf.name_scope('optimizer'):
      self.cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.out_layer) +
        self.l2_regularizer_beta * tf.nn.l2_loss(self.params['W_fc1']) +
        self.l2_regularizer_beta * tf.nn.l2_loss(self.params['W_nn1']))
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)

      self.test_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.out_layer, 1))
      self.test_accuracy = tf.reduce_mean(tf.cast(self.test_prediction, tf.float32))

    with tf.name_scope('summaries'):
      summaries = ['train_accuracy', 'train_cost', 'validate_accuracy', 'validate_cost', 'learning_rate']
      self.summary_scalars = dict()
      self.summary_objs = dict()
      for s in summaries:
        self.summary_scalars[s] = tf.placeholder('float32', None, name='scalar_' + s)
        self.summary_objs[s] = tf.summary.scalar(s, self.summary_scalars[s])

    # Save variables
    self.saver = tf.train.Saver(self.params)

    # Output summaries
    self.merged_summaries = tf.summary.merge_all()
    self.writer = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)
    tf.global_variables_initializer().run(session=self.sess)

  def train(self):
    per_batch = 32
    epochs = 300
    eval_per_n_batches = int(self.data.num_train_data_entries() / per_batch)

    avg_cost, avg_accu = 0, 0
    lowest_val_avg_cost, non_improving_count = 10000.0, 0

    for i in xrange(epochs * eval_per_n_batches):
      lr = Util.decaying_learning_rate(
        self.min_learning_rate,
        self.max_learning_rate,
        (self.decay_speed * eval_per_n_batches), i)
      xfeed, yfeed = self.data.next_batch_train(per_batch)
      _, train_cost, train_accuracy = self.sess.run(
        [self.optimizer, self.cross_entropy, self.test_accuracy], feed_dict={
        self.x : xfeed,
        self.y : yfeed,
        self.prob_keep_conv1 : 0.9,
        self.prob_keep_conv2 : 0.75,
        self.prob_keep_conv3 : 0.75,
        self.prob_keep_fc1   : 0.5,
        self.learning_rate : lr
      })
      avg_cost += train_cost
      avg_accu += train_accuracy

      if i % eval_per_n_batches == eval_per_n_batches - 1:
        xfeed, yfeed = self.data.validate_data_all()
        va, val_avg_cost = self.sess.run([self.test_accuracy, self.cross_entropy], feed_dict={
          self.x: xfeed,
          self.y: yfeed,
          self.prob_keep_conv1 : 1.0,
          self.prob_keep_conv2 : 1.0,
          self.prob_keep_conv3 : 1.0,
          self.prob_keep_fc1   : 1.0
        })
        self.__write_summary({
          'train_accuracy' : avg_accu / eval_per_n_batches,
          'validate_accuracy' : va,
          'train_cost' : avg_cost / eval_per_n_batches,
          'validate_cost' : val_avg_cost,
          'learning_rate' : lr
        }, int(i / float(eval_per_n_batches)))
        if val_avg_cost < lowest_val_avg_cost:
          lowest_val_avg_cost = val_avg_cost
          non_improving_count = 0
          self.saver.save(self.sess, os.path.join(dir_path, '../model.nosync/cnn_early_stopping_without_conc'))
        else:
          non_improving_count += 1
          print 'No improvement for past %d epochs' % non_improving_count
          if non_improving_count == 8:
            break
        avg_cost = 0
        avg_accu = 0

  def test(self):
    test_data = self.data.test_data_all()
    print self.sess.run(self.test_accuracy, feed_dict={
      self.x: test_data[0],
      self.y: test_data[1],
      self.prob_keep_conv1 : 1.0,
      self.prob_keep_conv2 : 1.0,
      self.prob_keep_conv3 : 1.0,
      self.prob_keep_fc1   : 1.0
    })
