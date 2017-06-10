import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio

from nets import nets_factory
from preprocessing import preprocessing_factory
import dataset_utils
import dataset_factory
import my_exceptions
import mypreproc.convert_cifar as cifar
import mypreproc.convert_voc as voc

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'model', 'inception_resnet_v2', 'Model name to use')

tf.app.flags.DEFINE_string(
    'scope', 'InceptionResnetV2', 'variable scope to access to logits')

tf.app.flags.DEFINE_string(
    'dataset_name', 'cifar10',
    'The source dataset name')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/home/cideep/Work/tensorflow/datasets/cifar10/tfrecord',
    'The source dataset name')

tf.app.flags.DEFINE_string(
    'checkpoint_dir', '/home/cideep/Work/tensorflow/checkpoints/my-fine-tuned/inc-resnet-v2-cifar10/model.ckpt-50000',
    'The absolute filepath to a checkpoint file.')

tf.app.flags.DEFINE_string(
    'split', 'test',
    'The data split name')

tf.app.flags.DEFINE_string(
    'output_dir', '/home/cideep/Work/tensorflow/output-data/tmp',
    'Dir to write output data')

tf.app.flags.DEFINE_integer(
    'batch_size', 32,
    'batch input size')


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


def create_inputs(filename, batch_size, image_size):
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
    print("filename", filename)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
        # Even when reading in multiple threads, share the filename queue.
        image, label = read_and_decode(filename_queue)

        # pre process for input
        image_preprocessing_fn = preprocessing_factory.get_preprocessing( \
            FLAGS.model, is_training=False)
        processed_image = image_preprocessing_fn(image, image_size, image_size)
        processed_images, raw_images, labels = tf.train.batch( \
            tensors=[processed_image, image, label], \
            batch_size=batch_size, num_threads=2, capacity=1000)

    print("input shape", processed_images.shape)
    return processed_images, raw_images, labels


def pass_network(processed_images, model_name, network_fn):
    arg_scope = nets_factory.arg_scopes_map[model_name]
    with slim.arg_scope(arg_scope()):
        logits, _ = network_fn(processed_images)
    probabilities = tf.nn.softmax(logits)
    return probabilities


def init_ops(sess):
    init_fn = slim.assign_from_checkpoint_fn(
        tf.train.latest_checkpoint(FLAGS.checkpoint_dir),
        slim.get_model_variables(FLAGS.scope))
    init_fn(sess)
    init_local = tf.local_variables_initializer()
    sess.run(init_local)


def print_results(images, labels, probabilities, class_names):
    probabilities = np.squeeze(probabilities)

    prob_first = probabilities[0, 0:]
    print('first sample of batch, label and probs:', labels[0], '\n', prob_first)
    sorted_inds = [i[0] for i in sorted(enumerate(-prob_first), key=lambda x: x[1])]
    result_text = '%s => %.2f' % (class_names[sorted_inds[0]], 100 * prob_first[sorted_inds[0]])

    disp_image = images[0]
    plt.clf()
    plt.imshow(disp_image.astype(np.uint8))
    plt.text(disp_image.shape[0], disp_image.shape[1]/2, result_text,
             horizontalalignment='left', verticalalignment='center',
             fontsize=15, color='blue')
    plt.axis('off')
    plt.pause(.1)
    plt.draw()


def stack_results(output_data, labels, probabilities, step, bs):
    if (step+1) * bs >= output_data.shape[0]:
        raise my_exceptions.GeneralError('out of data rows at %d' % step)
    output_data[(step * bs):((step+1) * bs), 0:] = np.column_stack((labels, probabilities))
    return output_data


def save_probabilities(output_data, dstpath, filename):
    # if os.path.exists(dstpath):
    #     shutil.rmtree(dstpath, ignore_errors=True)
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    output_path = os.path.join(dstpath, filename)
    np.save(output_path, output_data)
    matfile_path = output_path + '.mat'
    sio.savemat(matfile_path, {"data": output_data})


def main(_):
    print("dataset dir", FLAGS.dataset_dir)
    class_names = dataset_utils.read_label_file(FLAGS.dataset_dir)
    filename = os.path.join(FLAGS.dataset_dir, '%s.tfrecord' % FLAGS.split)

    splits_to_sizes, num_classes, image_shape, items_to_descriptions \
        = dataset_factory.dataset_config(FLAGS.dataset_name)

    output_data = np.zeros((splits_to_sizes[FLAGS.split],num_classes+1))
    print(output_data.shape)

    with tf.Graph().as_default():
        network_fn = nets_factory.get_network_fn(
            FLAGS.model, num_classes=num_classes, is_training=False)
        processed_images, raw_images, labels = create_inputs( \
            filename, FLAGS.batch_size, network_fn.default_image_size)
        probabilities = pass_network(processed_images, FLAGS.model, network_fn)

        sess = tf.Session()
        init_ops(sess)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        plt.figure()
        np.set_printoptions(precision=3, suppress=True)

        try:
            step = 0
            while not coord.should_stop():
                [np_images, np_labels, np_probabilities] = sess.run(
                    [raw_images, labels, probabilities])

                # print_results(np_images, np_labels, np_probabilities, class_names)
                output_data = stack_results(output_data, np_labels, np_probabilities, step, FLAGS.batch_size)
                step += 1
                if step%100==0:
                    print("step:", step)

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

    save_probabilities(output_data, FLAGS.output_dir, FLAGS.split)


if __name__ == '__main__':
    tf.app.run()
