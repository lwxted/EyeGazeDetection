#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from PIL import Image

class EyeRegionData(object):
  def __init__(self):
    super(EyeRegionData, self).__init__()
    self.x = None
    self.y = None

    self.train_upto = 0
    self.validate_upto = 0
    self.test_upto = 0

    self.train_batch_count = 0

  def load_data_from_file(self, full_npy):
    full = np.load(full_npy)

    self.x = full[..., :-1]
    raw_y = full[..., -1].tolist()
    self.y = []
    for i in xrange(len(raw_y)):
      self.y.append([1.0 if raw_y[i] == j else 0.0 for j in xrange(7)])
    self.y = np.asarray(self.y, dtype=np.float32)

    # Normalize x by subtracting mean image
    self.x -= np.mean(self.x, axis=0)

    self.train_upto = full.shape[0] * 7 / 10
    self.validate_upto = self.train_upto + full.shape[0] * 2 / 10
    self.test_upto = full.shape[0]

  def next_batch_train(self, batch_size):
    terminal_batch_count = self.train_batch_count + batch_size
    x = self.x[self.train_batch_count:terminal_batch_count]
    y = self.y[self.train_batch_count:terminal_batch_count]

    if terminal_batch_count <= self.train_upto:
      self.train_batch_count = terminal_batch_count
      return x, y

    rotated_batch_count = batch_size - x.shape[0]
    x = np.concatenate((x, self.x[:rotated_batch_count]), axis=0)
    y = np.concatenate((y, self.y[:rotated_batch_count]), axis=0)
    self.train_batch_count = rotated_batch_count
    return x, y

  def test_data_all(self):
    return self.x[self.validate_upto:self.test_upto], \
           self.y[self.validate_upto:self.test_upto]

  def validate_data_all(self):
    return self.x[self.train_upto:self.validate_upto], \
           self.y[self.train_upto:self.validate_upto]

  def num_train_data_entries(self):
    return self.train_upto
