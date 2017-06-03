from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os
import fnmatch
import numpy as np
import tensorflow as tf
from my_exceptions import PathError

IMAGE_SIZE = 32

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _collect_data(dataset_dir, file_pattern, label_key):
  src_files = fnmatch.filter(os.listdir(dataset_dir), file_pattern)
  if len(src_files)==0:
    raise PathError('no data file such as %s in %s' % (file_pattern, dataset_dir))

  total_images = np.array([])
  total_labels = np.array([])

  for filename in src_files:
    filename = os.path.join(dataset_dir, filename)
    print('   collect_data/srcfile name:', filename)
    with tf.gfile.Open(filename, 'rb') as f:
      data = pickle.load(f, encoding='bytes')
    images = data[b'data']
    labels = np.asarray(data[label_key])
    print('   collect_data/image_shape:', images.shape, 'label_shape:', labels.shape)

    if total_images.size == 0:
      total_images = images
    else:
      total_images = np.concatenate((total_images, images), axis=0)

    if total_labels.size == 0:
      total_labels = labels
    else:
      total_labels = np.concatenate((total_labels, labels), axis=0)

  if total_images.shape[0]==0 or total_images.shape[0]!=total_labels.shape[0]:
    raise PathError('invalid image or label size: %d != %d' \
                    % (total_images.shape[0], total_labels.shape[0]))
  print('collect_data/total_image_shape:', total_images.shape,
        'total_label_shape:', total_labels.shape)
  return total_images, total_labels


def _shuffle_and_split_data(images, labels, split_ratio):
  num_images = images.shape[0]
  shuffled_indices = np.arange(num_images)
  np.random.shuffle(shuffled_indices)
  images = images[shuffled_indices,:]
  labels = labels[shuffled_indices]
  num_split = int(num_images*split_ratio)
  return images[0:num_split,:], labels[0:num_split], \
         images[num_split:num_images,:], labels[num_split:num_images]


def _write_tfrecord(images, labels, output_filename):
  with tf.python_io.TFRecordWriter(output_filename) as writer:
    num_images = images.shape[0]
    images = images.reshape((num_images, 3, IMAGE_SIZE, IMAGE_SIZE))
    images = images.transpose((0,2,3,1))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]
    print('   write_tfrecord/image_shape:', images.shape, 'label_shape:', labels.shape)

    for index in range(num_images):
      image_raw = images[index].tostring()
      example = tf.train.Example(features=tf.train.Features(feature={
          'image/encoded': _bytes_feature(image_raw),
          'image/format': _bytes_feature(b'raw'),
          'image/class/label': _int64_feature(int(labels[index])),
          'image/height': _int64_feature(rows),
          'image/width': _int64_feature(cols),
          'image/depth': _int64_feature(depth)
          }))
      writer.write(example.SerializeToString())
    print('write_tfrecord/data was written in %s' % output_filename)


def _read_labelfile(dataset_dir, label_pattern, labelname_key):
  label_files = fnmatch.filter(os.listdir(dataset_dir), label_pattern)
  if len(label_files)==0:
    raise PathError('no label file in %s' % dataset_dir)
  label_filename = os.path.join(dataset_dir, label_files[0])
  with tf.gfile.Open(label_filename, 'rb') as f:
    data = pickle.load(f, encoding='bytes')
    labels_bytes = data[labelname_key]
    labels_str = [label.decode('utf-8') for label in labels_bytes]
    return dict(zip(list(range(len(labels_str))), labels_str))


def _write_labels(labels, label_filename):
  with tf.gfile.Open(label_filename, 'w') as f:
    for index, labelname in labels.items():
      f.write('%d:%s\n' % (index, labelname))
    print('labels are written in', label_filename)


def convert_cifar(dataset_dir, validation_ratio, \
                  training_pattern, test_pattern, label_pattern, label_key, labelname_key):
  print('Read cifar10 dataset and convert it to .tfrecord format')
  # check paths
  if not tf.gfile.Exists(dataset_dir):
    raise PathError('dataset dir [%s] does not exists' % dataset_dir)
  output_dir = '%s/tfrecord' % dataset_dir
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  # write the labels file
  class_names = _read_labelfile(dataset_dir, label_pattern, labelname_key)
  label_filename = os.path.join(output_dir, 'labels.txt')
  _write_labels(class_names, label_filename)
  num_classes = len(class_names)

  # process the training and validation data
  images, labels = _collect_data(dataset_dir, training_pattern, label_key)
  training_images, training_labels, validation_images, validataion_labels \
      = _shuffle_and_split_data(images, labels, validation_ratio)

  training_record_name = os.path.join(output_dir, 'train.tfrecord')
  _write_tfrecord(training_images, training_labels, training_record_name)

  validation_record_name = os.path.join(output_dir, 'validation.tfrecord')
  _write_tfrecord(validation_images, validataion_labels, validation_record_name)

  # process the test data
  test_images, test_labels = _collect_data(dataset_dir, test_pattern, label_key)
  test_record_name = os.path.join(output_dir, 'test.tfrecord')
  _write_tfrecord(test_images, test_labels, test_record_name)

  print('Finished converting the cifar%d dataset!' % num_classes)


def convert_cifar10(dataset_dir, validation_ratio):
  convert_cifar(dataset_dir, validation_ratio, \
                'data_batch*', 'test_batch*', '*.meta', \
                b'labels', b'label_names')


def convert_cifar100(dataset_dir, validation_ratio):
  convert_cifar(dataset_dir, validation_ratio, \
                'train', 'test', 'meta', \
                b'fine_labels', b'fine_label_names')
