import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from nets import inception
from preprocessing import inception_preprocessing
import convert_cifar as cifar
import dataset_utils

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'model_name', 'inception_resnet_v2', 'Model name to use')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/home/cideep/Work/tensorflow/checkpoints/my-fine-tuned/inc-resnet-v2-cifar10/model.ckpt-50000',
    'The absolute filepath to a checkpoint file.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'cifar10',
    'The source dataset name')

tf.app.flags.DEFINE_integer(
    'num_classes', 10,
    'Number of output classes.')

tf.app.flags.DEFINE_integer(
    'batch_size', 2,
    'batch input size')

FLAGS = tf.app.flags.FLAGS


name_to_model_net = {'inception_v4': inception.inception_v4,
                     'inception_resnet_v2': inception.inception_resnet_v2}
name_to_arg_scope = {'inception_v4': inception.inception_v4_arg_scope,
                     'inception_resnet_v2': inception.inception_resnet_v2_arg_scope}
name_to_name_scope = {'inception_v4': 'InceptionV4',
                     'inception_resnet_v2': 'InceptionResnetV2'}
name_to_image_size = {'inception_v4': inception.inception_v4.default_image_size,
                     'inception_resnet_v2': inception.inception_resnet_v2.default_image_size}

DATASET_PATH_PATTERN = '/home/cideep/Work/tensorflow/datasets/%s/tfrecord'



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
    image = tf.reshape(flat_image, tf.constant([cifar.IMAGE_SIZE, cifar.IMAGE_SIZE, 3], dtype=tf.int32))

    return image, label


def inputs(batch_size):
    """Reads input data num_epochs times.

    Args:
    batch_size: Number of examples per returned batch.

    Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
    """
    filename = (DATASET_PATH_PATTERN % FLAGS.dataset_name) + '/test.tfrecord'

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
        # Even when reading in multiple threads, share the filename queue.
        image, label = read_and_decode(filename_queue)

        # pre process for inception input
        image_size = name_to_image_size[FLAGS.model_name]
        processed_image = inception_preprocessing.preprocess_image(
            image, image_size, image_size, is_training=False)

        processed_images, raw_images, labels = tf.train.batch(tensors=[processed_image, image, label],
                                                  batch_size=batch_size,
                                                  num_threads=2, capacity=1000)

    print(processed_images.shape)
    return processed_images, raw_images, labels


def pass_network(processed_images, model_name):
    # Create the model, use the default arg scope to configure the batch norm parameters.
    net_logit = name_to_model_net[model_name]
    arg_scope = name_to_arg_scope[model_name]
    with slim.arg_scope(arg_scope()):
        logits, _ = net_logit(processed_images, num_classes=FLAGS.num_classes, is_training=False)
    probabilities = tf.nn.softmax(logits)
    return probabilities


def print_results(images, labels, probabilities, class_names):
    probabilities = np.squeeze(probabilities)
    print('probabilities of', labels, '\n', probabilities)

    prob_first = probabilities[0, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-prob_first), key=lambda x: x[1])]
    result_text = '%s => %.2f' % (class_names[sorted_inds[0]], 100 * prob_first[sorted_inds[0]])

    disp_image = images[0]
    plt.clf()
    plt.imshow(disp_image.astype(np.uint8))
    plt.text(disp_image.shape[0], disp_image.shape[1]/2, result_text,
             horizontalalignment='left', verticalalignment='center',
             fontsize=15, color='blue')
    plt.axis('off')
    plt.pause(1)
    plt.draw()


def main(_):
    class_names = dataset_utils.read_label_file(DATASET_PATH_PATTERN % FLAGS.dataset_name)

    with tf.Graph().as_default():
        processed_images, raw_images, labels = inputs(batch_size=FLAGS.batch_size)

        probabilities = pass_network(processed_images, FLAGS.model_name)

        sess = tf.Session()
        init_fn = slim.assign_from_checkpoint_fn(
            FLAGS.checkpoint_path,
            slim.get_model_variables(name_to_name_scope[FLAGS.model_name]))
        init_fn(sess)
        init_local = tf.local_variables_initializer()
        sess.run(init_local)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        plt.figure()
        np.set_printoptions(precision=3, suppress=True)

        try:
            step = 0
            while not coord.should_stop():
                step += 1
                [np_images, np_labels, np_probabilities] = sess.run(
                    [raw_images, labels, probabilities])

                print_results(np_images, np_labels, np_probabilities, class_names)

        except tf.errors.OutOfRangeError:
            print('Done evaluating for %d steps.' % step)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    tf.app.run()
