import os
import argparse
import sys
import numpy as np
import pandas as pd
import convert_voc_annots_df as vocdf
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append("/home/cideep/Work/tensorflow/codes")
import nets.inception_v4 as inceptionV4

FLAGS = None
df_cols = vocdf.df_cols
image_size = inceptionV4.inception_v4.default_image_size


def read_annotations():
  df_file_name = '%s/%s/%s' % (FLAGS.dataset_dir, FLAGS.annot_dir, FLAGS.annot_dir) + '.csv'
  print(df_file_name)
  df = pd.read_csv(df_file_name, sep='\t')
  print("data length", len(df))
  return df


def create_labels(annots):
  label_names = annots.category.unique()
  label_map = dict(zip(label_names, list(range(len(label_names)))))
  print('label map', label_map)
  len_annots = len(annots)
  labels = np.zeros([len_annots,1])

  for idx in range(len_annots):
    catgname = annots.ix[idx, 'category']
    labels[idx] = label_map[catgname]
  return labels


def shuffle_and_split_annots(annots, split_ratio):
  length = len(annots)
  to_train = int(length * split_ratio[0])
  to_val = int(length * (split_ratio[0] + split_ratio[1]))
  shuffled_indices = np.arange(length)
  np.random.shuffle(shuffled_indices)

  train_inds = shuffled_indices[range(0, to_train)]
  val_inds = shuffled_indices[range(to_train, to_val)]
  test_inds = shuffled_indices[range(to_val, length)]
  # train_inds = shuffled_indices[range(0, 100)]
  # val_inds = shuffled_indices[range(100, 200)]
  # test_inds = shuffled_indices[range(200, 300)]

  return [annots.ix[train_inds,df_cols].reset_index(), \
          annots.ix[val_inds,df_cols].reset_index(),
          annots.ix[test_inds,df_cols].reset_index()]


def collect_images(annots):
  image_dir = '%s/%s' % (FLAGS.dataset_dir, FLAGS.image_dir)
  num_imgs = len(annots)
  total_images = np.zeros([num_imgs, image_size, image_size, 3], dtype=np.uint8)

  for idx in range(len(annots)):
    img = read_and_preproc(os.path.join(image_dir, annots.ix[idx,'image_name']),
                           annots.ix[idx, ['xmin', 'xmax', 'ymin', 'ymax']].tolist())
    total_images[idx] = img
    modval = int(num_imgs/100);
    implot = None
    if idx % modval == 0:
      print('img', idx, '\t', annots.ix[idx,['category','image_name']].tolist())
      if implot is None:
        implot = plt.imshow(img)
      else:
        implot.set_data(img)
      plt.pause(.1)
      plt.draw()

  print("collecting images is finished")
  return total_images


def read_and_preproc(image_path, bndbox):
  img = io.imread(image_path)
  img = img[bndbox[2]:bndbox[3], bndbox[0]:bndbox[1]]
  img = transform.resize(img, [image_size, image_size], mode='reflect') * 255
  img = img.astype(np.uint8)
  return img


def write_tfrecord(record_name, images, labels):
  with tf.python_io.TFRecordWriter(record_name) as writer:
    num_images = images.shape[0]
    depth = images.shape[3]
    print('   write_tfrecord/image_shape:', images.shape, 'label_shape:', len(labels))

    for index in range(num_images):
      image_raw = images[index].tostring()
      example = tf.train.Example(features=tf.train.Features(feature={
          'image/encoded': bytes_feature(image_raw),
          'image/format': bytes_feature(b'raw'),
          'image/class/label': int64_feature(int(labels[index])),
          'image/height': int64_feature(image_size),
          'image/width': int64_feature(image_size),
          'image/depth': int64_feature(depth)
          }))
      writer.write(example.SerializeToString())
    print('tfrecord was written in %s\n\n' % record_name)


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_labels(annots, label_filename):
  label_names = annots.category.unique()
  labels = dict(zip(list(range(len(label_names))), label_names))
  with tf.gfile.Open(label_filename, 'w') as f:
    for index, labelname in labels.items():
      f.write('%d:%s\n' % (index, labelname))
    print('labels are written in', label_filename)


def main():
  annots = read_annotations()
  split_ratio = [0.7, 0.15, 0.15]
  annot_splits = shuffle_and_split_annots(annots, split_ratio)
  split_names = ['train', 'validation', 'test']
  output_dir = "%s/%s" % (FLAGS.dataset_dir, FLAGS.record_dir)
  print("split sizes", len(annot_splits[0]), len(annot_splits[1]), len(annot_splits[2]))

  label_filename = os.path.join(output_dir, 'labels.txt')
  write_labels(annot_splits[0], label_filename)


  for i in range(3):
    labels = create_labels(annot_splits[i])
    images = collect_images(annot_splits[i])
    record_name = os.path.join(output_dir, '%s.tfrecord' % split_names[i])
    write_tfrecord(record_name, images, labels)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--dataset_dir',
    type=str,
    default='/home/cideep/Work/tensorflow/datasets/Link-to-datasets/voc2012/VOC-2012-train',
    help='dataset dir'
  )
  parser.add_argument(
    '--image_dir',
    type=str,
    default='JPEGImages',
    help='dir of original images in jpeg files'
  )
  parser.add_argument(
    '--annot_dir',
    type=str,
    default='AnnotationsDF',
    help='dir of annotations in csv dataframe files'
  )
  parser.add_argument(
    '--record_dir',
    type=str,
    default='tfrecord',
    help='tfrecord formatted output file'
  )
  FLAGS, unparsed = parser.parse_known_args()
  main()
