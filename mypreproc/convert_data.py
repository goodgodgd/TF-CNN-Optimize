from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import fnmatch
import tensorflow as tf
from my_exceptions import PathError
from convert_cifar import convert_cifar10, convert_cifar100

FLAGS = None

def convert_data():
  if FLAGS.dataset == 'cifar10':
    dataset_dir = '/home/cideep/Work/tensorflow/datasets/cifar-10'
    convert_cifar10(dataset_dir, FLAGS.validation_ratio)
  elif FLAGS.dataset == 'cifar100':
    dataset_dir = '/home/cideep/Work/tensorflow/datasets/cifar-100'
    convert_cifar100(dataset_dir, FLAGS.validation_ratio)
  # elif FLAGS.dataset == 'voc2012':
  #   convert_voc2012(FLAGS.input_dir, FLAGS.validation_ratio)
  else:
    raise PathError('%s: Not supported dataset' % FLAGS.input_dir)


def main(_):
  try:
    convert_data()
  except PathError as pe:
    print(pe)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    help="""possible options: 'cifar10', 'cifar100', 'voc2012'"""
  )
  parser.add_argument(
    '--validation_ratio',
    type=float,
    default=0.2,
    help='ratio of validation data from training data'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
