import os
import numpy as np
import my_exceptions
import argparse

FLAGS = None
np.set_printoptions(precision=3, suppress=True)


def load_data(data_dir, split_name):
  filename = os.path.join(data_dir, split_name+'.npy')
  assert os.path.exists(filename)

  data = np.load(filename)
  labels = data[0:,0]
  probs = data[0:,1:]
  probs = probs.T
  print('loaded data shape:', labels.shape, probs.shape)
  return labels.astype(int), probs


def evaluate_accuracy(labels, probs):
  predict = np.argmax(probs, axis=0)
  assert len(labels)==len(predict)
  match_count = np.sum(predict==labels)
  accuracy = match_count / len(labels)
  return accuracy

def evaluate_each_class(labels, probs):
  num_classes = probs.shape[0]
  accuracies = np.zeros(num_classes)
  for i in range(num_classes):
    inds = labels==i
    accuracies[i] = evaluate_accuracy(labels[inds], probs[0:,inds])
  return accuracies


def convert_to_one_hot(vector, num_classes=None):
  assert len(vector) > 0
  if num_classes is None:
    num_classes = (np.max(vector) + 1).astype(int)
  assert num_classes > np.max(vector)

  result = np.zeros(shape=(num_classes, len(vector)))
  result[vector, np.arange(len(vector))] = 1
  return result.astype(int)


def inverse_matrix(column_long_matrix):
  mat1 = column_long_matrix
  mat2 = np.dot(mat1, mat1.T)
  mat3 = np.linalg.inv(mat2)
  mat4 = np.dot(mat1.T, mat3)
  return mat4


def total_optimal_weight(labels, probs):
  probs_padded = np.vstack((probs, np.ones(probs.shape[1],dtype=np.float)))
  # print('probs padded\n', probs_padded[0:, 0:10])
  probs_pinv = inverse_matrix(probs_padded)
  H_cols = probs.shape[0] + 1
  assert np.allclose(np.eye(H_cols,H_cols), np.dot(probs_padded, probs_pinv))

  probs_target = convert_to_one_hot(labels)
  H = np.dot(probs_target, probs_pinv)
  # H = np.eye(probs.shape[0], probs.shape[0]+1)
  return H


def calc_class_weight(labels, num_classes):
  data_len = len(labels)
  class_count = [np.sum(labels==i) for i in range(num_classes)]
  count_inv = np.divide(1., class_count)
  class_weights = np.array([count_inv[i]/np.sum(count_inv) for i in range(num_classes)])
  S = np.zeros([data_len, data_len])
  S[np.arange(data_len), np.arange(data_len)] = class_weights[labels]
  return S


def average_optimal_weight(labels, probs):
  num_classes = probs.shape[0]
  S = calc_class_weight(labels, num_classes)
  probs_padded = np.vstack((probs, np.ones(probs.shape[1], dtype=np.float)))
  probs_target = convert_to_one_hot(labels)
  mat1 = np.dot(probs_target, S)
  mat2 = np.dot(probs_padded, S)
  H = np.dot(mat1, inverse_matrix(mat2))
  return H


def evaluate_result(labels, probs, H):
  print('\noptimal weight bias\n', H)

  probs_target = convert_to_one_hot(labels)
  probs_padded = np.vstack((probs, np.ones(probs.shape[1], dtype=np.float)))
  probs_opt = np.dot(H, probs_padded)

  raw_error = np.sum(np.abs(np.subtract(probs_target, probs)))
  opt_error = np.sum(np.abs(np.subtract(probs_target, probs_opt)))
  raw_opt_diff = np.sum([np.abs(np.subtract(probs, probs_opt)) > 0.2])
  print('prob error raw, opt, diffcount: %.2f, %.2f, %d' % (raw_error, opt_error, raw_opt_diff))

  accuracy_raw = evaluate_each_class(labels, probs)
  accuracy_opt = evaluate_each_class(labels, probs_opt)
  print('total test accuracies (opt)', evaluate_accuracy(labels, probs_opt))
  print('test per class accuracies (opt)\n', accuracy_opt)
  print('per class accuracy diff\n', np.subtract(accuracy_opt, accuracy_raw))
  print('\n')


def main():
  # each column of probs_* corresponds to an output of CNN
  labels_vali, probs_vali = load_data(FLAGS.data_dir, 'validation')
  labels_test, probs_test = load_data(FLAGS.data_dir, 'test')

  vali_acc_raw = evaluate_each_class(labels_vali, probs_vali)
  test_acc_raw = evaluate_each_class(labels_test, probs_test)
  print('vali, test total accuracies (raw)', evaluate_accuracy(labels_vali, probs_vali), \
        evaluate_accuracy(labels_test, probs_test))
  print('vali, test per class accuracies (raw)\n', vali_acc_raw, '\n', test_acc_raw)

  H = total_optimal_weight(labels_test, probs_test)
  evaluate_result(labels_test, probs_test, H)

  H = average_optimal_weight(labels_vali, probs_vali)
  evaluate_result(labels_test, probs_test, H)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--data_dir',
    type=str,
    default='/home/cideep/Work/tensorflow/output-data/inception_v4_cifar10',
    help='source data dir, labels and probabilities '
  )
  FLAGS, unparsed = parser.parse_known_args()
  main()
