#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from scipy.misc import imresize

import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def rgb2gray(rgb):
  return np.round(np.dot(rgb[...,:3], [0.299, 0.587, 0.114]))

def main():
  all_data_left, all_labels_left = [], []
  all_data_right, all_labels_right = [], []

  base_data_dir_path = os.path.join(dir_path, '../../data_eye.nosync/')
  base_data_processed_dir_path = os.path.join(dir_path, '../../data_processed.nosync/')

  labels = {
    0 : '00.Centre',
    1 : '01.UpRight',
    2 : '02.UpLeft',
    3 : '03.Right',
    4 : '04.Left',
    5 : '05.DownRight',
    6 : '06.DownLeft'
  }

  for num_label, label in labels.iteritems():
    label_dir_path = os.path.join(base_data_dir_path, label)
    for image_name in os.listdir(label_dir_path):
      if '.jpg' not in image_name:
        continue
      image_path = os.path.join(label_dir_path, image_name)
      im = Image.open(image_path)
      imarray = np.array(im)
      imarray = rgb2gray(imarray)
      imarray = imresize(imarray, (42, 50), 'lanczos')

      if '_0.jpg' in image_name:
        all_data_left.append(imarray)
        all_labels_left.append(num_label)
      elif '_1.jpg' in image_name:
        all_data_right.append(imarray)
        all_labels_right.append(num_label)

  blob_left_data = np.asarray(all_data_left, dtype=np.float32)
  blob_left_labels = np.asarray(all_labels_left, dtype=np.float32)
  blob_right_data = np.asarray(all_data_right, dtype=np.float32)
  blob_right_labels = np.asarray(all_labels_right, dtype=np.float32)

  print blob_left_data.shape
  print blob_left_labels.shape
  print blob_right_data.shape
  print blob_right_labels.shape

  if not os.path.exists(base_data_processed_dir_path):
    os.makedirs(base_data_processed_dir_path)

  np.save(os.path.join(base_data_processed_dir_path, 'left_data.npy'), blob_left_data)
  np.save(os.path.join(base_data_processed_dir_path, 'left_labels.npy'), blob_left_labels)
  np.save(os.path.join(base_data_processed_dir_path, 'right_data.npy'), blob_right_data)
  np.save(os.path.join(base_data_processed_dir_path, 'right_labels.npy'), blob_right_labels)

  # Shuffled
  x_left = np.reshape(blob_left_data, (blob_left_data.shape[0], 42 * 50))
  y_left = np.reshape(blob_left_labels, (blob_left_data.shape[0], 1))

  x_right = np.reshape(blob_right_data, (blob_right_data.shape[0], 42 * 50))
  y_right = np.reshape(blob_right_labels, (blob_right_data.shape[0], 1))

  full_data_left = np.hstack((x_left, y_left))
  full_data_right = np.hstack((x_right, y_right))

  np.random.shuffle(full_data_left)
  np.random.shuffle(full_data_right)

  np.save(os.path.join(base_data_processed_dir_path, 'left_full_data_shuffled.npy'), full_data_left)
  np.save(os.path.join(base_data_processed_dir_path, 'right_full_data_shuffled.npy'), full_data_right)

if __name__ == '__main__':
  main()
