#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from cnn.EyeRegionData import EyeRegionData
from cnn.ModelV1 import ModelV1

if __name__ == '__main__':
  dir_path = os.path.dirname(os.path.realpath(__file__))
  data = EyeRegionData()
  data.load_data_from_file(os.path.join(dir_path, '../data_processed.nosync/left_full_data_shuffled.npy'))
  model = ModelV1(data, os.path.join(dir_path, 'log.nosync/modelv1/run6'))
  model.train()
  model.test()
