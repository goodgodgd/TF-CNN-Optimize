
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import dataset_utils

slim = tf.contrib.slim

FILE_PATTERN = '%s_%s.tfrecord'


def dataset_config(dataset_name):
  if dataset_name == 'cifar10':
    splits_to_sizes = {'train': 40000, 'validatation': 10000, 'test': 10000}
    num_classes = 10
    example_shape = [32, 32, 3]
    items_to_descriptors = {
      'image': 'A [32 x 32 x 3] color image.',
      'label': 'A single integer between 0 and 9',
    }
  elif dataset_name == 'cifar100':
    splits_to_sizes = {'train': 40000, 'validatation': 10000, 'test': 10000}
    num_classes = 100
    example_shape = [32, 32, 3]
    items_to_descriptors = {
      'image': 'A [32 x 32 x 3] color image.',
      'label': 'A single integer between 0 and 99',
    }
  else:
    raise ValueError('dataset name %s was not recognized.' % dataset_name)

  return splits_to_sizes, num_classes, example_shape, items_to_descriptors


def get_dataset(dataset_name, split_name, dataset_dir):

  splits_to_sizes, num_classes, example_shape, items_to_descriptions \
    = dataset_config(dataset_name)

  if split_name not in splits_to_sizes:
    raise ValueError('split name %s was not recognized.' % split_name)

  data_file = os.path.join(dataset_dir, FILE_PATTERN % (dataset_name, split_name))

  # Allowing None in the signature so that dataset_factory can use the default.
  reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(shape=example_shape),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=data_file,
      reader=reader,
      decoder=decoder,
      num_samples=splits_to_sizes[split_name],
      items_to_descriptions=items_to_descriptions,
      num_classes=num_classes,
      labels_to_names=labels_to_names)
