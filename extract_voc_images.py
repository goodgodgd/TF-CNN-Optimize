import os
import argparse
import sys
import fnmatch
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import time


# def modify_image(image):
#     resized = tf.image.resize_images(image, 180, 180, 1)
#     resized.set_shape([180,180,3])
#     flipped_images = tf.image.flip_up_down(resized)
#     return flipped_images
#
#
# def inputs():
#     filenames = ['img1.jpg', 'img2.jpg' ]
#     filename_queue = tf.train.string_input_producer(filenames)
#     filename,read_input = read_image(filename_queue)
#     reshaped_image = modify_image(read_input)
#     return filename,reshaped_image
#
# with tf.Graph().as_default():
#     image = inputs()
#     init = tf.initialize_all_variables()
#     sess = tf.Session()
#     sess.run(init)
#     tf.train.start_queue_runners(sess=sess)
#     for i in xrange(10):
#         filename,img = sess.run(image)
#         print (filename)
#         img = Image.fromarray(img, "RGB")
#         img.save(os.path.join(cur_dir,"foo"+str(i)+".jpeg"))


def read_annotations():
  df_file_name = '%s/%s/%s' % (FLAGS.dataset_path, FLAGS.src_annot_dir, FLAGS.src_annot_dir) + '.csv'
  df = pd.read_csv(df_file_name, sep='\t')
  return df


def extract_specific_category(annots, category):
  annot_subset = annots.loc[annots['category']==category]
  return annot_subset.iloc[0:5,:]


def get_input_queue(annots):
  filenames = annots['image_name'].tolist()
  filenames = [os.path.join(FLAGS.dataset_path + '/' + FLAGS.src_image_dir, fname) for fname in filenames]
  xmins = annots['xmin'].tolist()
  xmaxs = annots['xmax'].tolist()
  ymins = annots['ymin'].tolist()
  ymaxs = annots['ymax'].tolist()
  widths  = annots['width'].tolist()
  heights = annots['height'].tolist()
  print('create input queue:', len(filenames))
  print(filenames[0])
  filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=1)
  input_queue = tf.train.slice_input_producer( \
                              [xmins, xmaxs, ymins, ymaxs, widths, heights], \
                              shuffle=False, num_epochs=1)
  return filename_queue, input_queue


def read_image(input_queue):
  filename_queue = input_queue[0]
  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)
  image = tf.image.decode_jpeg(value)
  return key, image


def crop_image(image, input_queue):
  offset_x = input_queue[0]
  offset_y = input_queue[2]
  width = input_queue[1] - input_queue[0]
  height = input_queue[3] - input_queue[2]
  # object_image = tf.image.resize_images(image,[height,width])
  # object_image = tf.image.crop_to_bounding_box(image=image, \
  #                              offset_height=offset_y, offset_width=offset_x, \
  #                              target_height=height, target_width=width)
  return image


def image_proc_tensor(annots):
  filename_queue, input_queue = get_input_queue(annots)
  filename_ts, image = read_image(filename_queue)
  object_image = crop_image(image, input_queue)
  return object_image


def run_and_save_image(image, category, num_images):
  # image = tf.image.resize_bilinear(image,[200,200])
  init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  sess = tf.Session()
  sess.run(init)
  tf.train.start_queue_runners(sess=sess)
  for i in range(1):
      img = sess.run(image)
      print('run_and_save_image:', img.shape)
      img = Image.fromarray(img, "RGB")
      img.show('image')
      # img.save(os.path.join(FLAGS.dataset_path, "foo"+str(i)+".jpeg"))




        # coord = tf.train.Coordinator()
  # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  # try:
  #   while not coord.should_stop():
  #     # Run training steps or whatever
  #     print(sess.run(image))
  #
  # except tf.errors.OutOfRangeError:
  #   print('Done training -- epoch limit reached')
  # finally:
  #   # When done, ask the threads to stop.
  #   coord.request_stop()
  #
  # # Wait for threads to finish.
  # coord.join(threads)
  # sess.close()

  # for i in range(num_images):
  #   filename,img = sess.run(image)
  #   print (filename)
  #   img = Image.fromarray(img, "RGB")
  #   img.save(os.path.join(cur_dir,"foo"+str(i)+".jpeg"))


def main(_):
  # with tf.device("/cpu:0"):
  annots = read_annotations()
  categories = annots.category.unique()
  print(categories)
  print(annots.columns)

  for category in categories:
    annot_subset = extract_specific_category(annots, category)
    subset_size = annot_subset.shape[0]
    image = image_proc_tensor(annot_subset)
    # filename_queue, boundbox_queue = get_input_queue(annot_subset)
    run_and_save_image(image, category, subset_size)


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
    default='AnnotationsDF',
    help='dir of dataframe annotation in csv file'
  )
  parser.add_argument(
    '--dst_image_dir',
    type=str,
    default='ObjectImages',
    help='dir of dataframe annotation in csv file'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
