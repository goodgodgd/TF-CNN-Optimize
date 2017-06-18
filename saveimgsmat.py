import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio

import dataset_factory
import my_exceptions
import mypreproc.convert_cifar as cifar
import mypreproc.convert_voc as voc

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'dataset_name', 'cifar100',
    'The source dataset name')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/home/cideep/Work/tensorflow/datasets/Link-to-datasets/cifar100/tfrecord',
    'The source dataset name')

tf.app.flags.DEFINE_string(
    'split', 'test',
    'The data split name')

tf.app.flags.DEFINE_string(
    'output_dir', '/home/cideep/Work/tensorflow/datasets/matimg',
    'Dir to write output data')


FLAGS = tf.app.flags.FLAGS

RECORD_IMAGE_SIZE = None
if "cifar" in FLAGS.dataset_name:
    RECORD_IMAGE_SIZE = cifar.IMAGE_SIZE
elif "voc" in FLAGS.dataset_name:
    RECORD_IMAGE_SIZE = voc.IMAGE_SIZE


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
          'image/encoded': tf.FixedLenFeature([], tf.string),
          'image/format': tf.FixedLenFeature([], tf.string),
          'image/class/label': tf.FixedLenFeature([], tf.int64),
          'image/height': tf.FixedLenFeature([], tf.int64),
          'image/width': tf.FixedLenFeature([], tf.int64),
          'image/depth': tf.FixedLenFeature([], tf.int64)
        })

    # Convert from a scalar string tensor to a 3D uint8 tensor
    flat_image = tf.decode_raw(features['image/encoded'], tf.uint8)
    label = tf.cast(features['image/class/label'], tf.int32)

    image = tf.reshape(flat_image, tf.constant([RECORD_IMAGE_SIZE, RECORD_IMAGE_SIZE, 3], dtype=tf.int32))

    return image, label


def create_inputs(filename):
    print("filename", filename)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
        # Even when reading in multiple threads, share the filename queue.
        image, label = read_and_decode(filename_queue)
    return image, label


def main(_):
    print("dataset dir", FLAGS.dataset_dir)
    filename = os.path.join(FLAGS.dataset_dir, '%s.tfrecord' % FLAGS.split)
    splits_to_sizes, num_classes, image_shape, items_to_descriptions \
        = dataset_factory.dataset_config(FLAGS.dataset_name)

    image_shape.insert(0, 5000)
    out_images = np.zeros(image_shape)
    out_labels = np.zeros(5000)
    print(out_images.shape)

    with tf.Graph().as_default():
        ts_image, ts_label = create_inputs(filename)

        sess = tf.Session()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        plt.figure()
        np.set_printoptions(precision=3, suppress=True)

        try:
            step = 0
            while not coord.should_stop():
                [np_image, np_label] = sess.run([ts_image, ts_label])
                out_images[step] = np_image
                out_labels[step] = np_label

                step += 1
                if step%100==0:
                    print("step:", step)
                if step >= out_images.shape[0]:
                  raise my_exceptions.GeneralError('out of data rows at %d' % step)

        except tf.errors.OutOfRangeError:
            print('Done evaluating for %d steps.' % step)
        except my_exceptions.GeneralError as ge:
            print('Done evaluating for %d steps.' % step, ge)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

    # save images and labels in mat
    out_images = out_images.astype(np.uint8)
    out_labels = out_labels.astype(np.uint16)
    filename = '%s/%s.mat' % (FLAGS.output_dir, FLAGS.dataset_name)
    print('output file:', filename)
    print('output shape:', out_images.shape)
    sio.savemat(filename, {"images": out_images, "labels": out_labels})


if __name__ == '__main__':
    tf.app.run()
