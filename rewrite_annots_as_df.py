import os
import argparse
import sys
import fnmatch
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as et
from my_exceptions import PathError

FLAGS = None
df_cols = ['category', 'image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'width', 'height']


def list_images(image_dir):
  imglist = fnmatch.filter(os.listdir(image_dir), '*.jpg')
  imglist.sort()
  return imglist


def convert_to_annot(annot_dir, image_file):
  annot_file = image_file.replace('jpg', 'xml')
  annot_file = os.path.join(annot_dir, annot_file)
  if tf.gfile.Exists(annot_file):
    return annot_file
  else:
    return ''


def read_each_annot(annot_file):
  doc = et.parse(annot_file)
  root = doc.getroot()
  objects = root.findall('object')
  categ_names = []
  bound_boxes = []

  size_node = root.find('size')
  width = int(size_node.findtext('width'))
  height= int(size_node.findtext('height'))
  image_size = [width, height]

  for obj in objects:
    try:
      name = obj.findtext('name')
      bndbox = obj.find('bndbox')
      bbox = [0, 0, 0, 0]
      bbox[0] = int(bndbox.findtext('xmin'))
      bbox[1] = int(bndbox.findtext('xmax'))
      bbox[2] = int(bndbox.findtext('ymin'))
      bbox[3] = int(bndbox.findtext('ymax'))
    except:
      print('exceptional annot in', annot_file, bndbox.findtext('xmin'), bndbox.findtext('xmax'), \
            bndbox.findtext('ymin'), bndbox.findtext('ymax'))
      continue
    categ_names.append(name)
    bound_boxes.append(bbox)
  return categ_names, bound_boxes, image_size


def create_dataframe(categ_names, image_name, bound_boxes, image_size):
  data = []
  for cname, bbox in zip(categ_names, bound_boxes):
    row_data = [cname, image_name]
    row_data.extend(bbox)
    row_data.extend(image_size)
    data.append(row_data)
  return pd.DataFrame(data, columns=df_cols)


def read_annotations(annot_dir, image_list):
  annot_data = pd.DataFrame()

  for image_file in image_list:
    annot_file = convert_to_annot(annot_dir, image_file)
    if not annot_file:
      print('image file [%s] does not have annotation' % image_file)
      continue

    categ_names, bound_boxes, image_size = read_each_annot(annot_file)
    new_df = create_dataframe(categ_names, image_file, bound_boxes, image_size)
    annot_data = annot_data.append(new_df, ignore_index=True)
  print('%s object annotations were loaded' % annot_data.shape[0])
  annot_data = annot_data.sort_values(by=['category', 'image_name'])
  print('annotations were sorted by category names')
  print(annot_data.head(5))
  return annot_data


def main(_):
  if not tf.gfile.Exists(FLAGS.dataset_path):
    raise PathError('dataset dir does not exists')

  image_dir = os.path.join(FLAGS.dataset_path, FLAGS.src_image_dir)
  image_list = list_images(image_dir)
  print('%d images are listed' % len(image_list))

  # annotations = DataFrame(category, filename, xmin, xmax, ymin, ymax)
  src_annot_path = os.path.join(FLAGS.dataset_path, FLAGS.src_annot_dir)
  annotations = read_annotations(src_annot_path, image_list)

  dst_annot_path = os.path.join(FLAGS.dataset_path, FLAGS.dst_annot_dir)
  if not tf.gfile.Exists(dst_annot_path):
    tf.gfile.MakeDirs(dst_annot_path)
  dst_annot_name = os.path.join(dst_annot_path, '%s.csv' % FLAGS.dst_annot_dir)
  annotations.to_csv(dst_annot_name, sep='\t')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--dataset_path',
    type=str,
    default='/home/cideep/Work/tensorflow/datasets/VOC-2012/VOC-2012-train',
    help='dataset dir'
  )
  parser.add_argument(
    '--src_image_dir',
    type=str,
    default='JPEGImages',
    help='dir of original images in jpeg files'
  )
  parser.add_argument(
    '--src_annot_dir',
    type=str,
    default='Annotations',
    help='dir of original annotations in xml files'
  )
  parser.add_argument(
    '--dst_annot_dir',
    type=str,
    default='AnnotationsDF',
    help='dir of dataframe annotation in csv file'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
