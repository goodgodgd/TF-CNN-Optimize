import numpy as np
import os
import tensorflow as tf
# import urllib2
from urllib.request import urlopen
import matplotlib.pyplot as plt

from nets import inception
from preprocessing import inception_preprocessing
import convert_cifar as cifar


slim = tf.contrib.slim
image_size = inception.inception_v3.default_image_size

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
    'batch_size', 32,
    'batch input size')

FLAGS = tf.app.flags.FLAGS


name_to_model_net = {'inception_v4': inception.inception_v4,
                     'inception_resnet_v2': inception.inception_resnet_v2}
name_to_arg_scope = {'inception_v4': inception.inception_v4_arg_scope,
                     'inception_resnet_v2': inception.inception_resnet_v2_arg_scope}
name_to_dataset = {'cifar10': '/home/cideep/Work/tensorflow/datasets/cifar-10/tfrecord/cifar10_test.tfrecord',
                   'cifar100': '/home/cideep/Work/tensorflow/datasets/cifar-100/tfrecord/cifar100_test.tfrecord'}
name_to_name_scope = {'inception_v4': 'InceptionV4',
                     'inception_resnet_v2': 'InceptionResnetV2'}
name_to_image_size = {'inception_v4': inception.inception_v4.default_image_size,
                     'inception_resnet_v2': inception.inception_resnet_v2.default_image_size}

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
    image_size = name_to_image_size[FLAGS.model_name]
    image = tf.reshape(flat_image, tf.constant([image_size, image_size, 3], dtype=tf.int32))

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
    filename = name_to_dataset[FLAGS.dataset_name]

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
        # Even when reading in multiple threads, share the filename queue.
        image, label = read_and_decode(filename_queue)

        # preprocess for inception input
        processed_image = inception_preprocessing.preprocess_image(
            image, cifar.IMAGE_SIZE, cifar.IMAGE_SIZE, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)
        labels = label

        # processed_images, labels = tf.train.batch(tensors=[processed_image, label],
        #                                           batch_size=batch_size,
        #                                           num_threads=2, capacity=1000)

    print(processed_images.shape)
    return processed_images, labels


def main(_):
    with tf.Graph().as_default():
        # images, labels = inputs(batch_size=FLAGS.batch_size)

        url = "http://static.trustedreviews.com/94/00003b9eb/f7f9/airplane.jpg"
        image_string = urlopen(url).read()
        image = tf.image.decode_jpeg(image_string, channels=3)
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        net_logit = name_to_model_net[FLAGS.model_name]
        arg_scope = name_to_arg_scope[FLAGS.model_name]
        with slim.arg_scope(arg_scope()):
            logits, _ = net_logit(processed_images, num_classes=FLAGS.num_classes, is_training=False)

        probabilities = tf.nn.softmax(logits)
        init_fn = slim.assign_from_checkpoint_fn(
            FLAGS.checkpoint_path,
            slim.get_model_variables(name_to_name_scope[FLAGS.model_name]))

        with tf.Session() as sess:
            init_fn(sess)
            np_images, np_probabilities = sess.run([image, probabilities])
            np_probabilities = np_probabilities[0, 0:]
            print('probabilities:', probabilities)
            sorted_inds = [i[0] for i in sorted(enumerate(-np_probabilities), key=lambda x:x[1])]
        # names = imagenet.create_readable_names_for_imagenet_labels()
        result_text=''
        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.2f%% => [%d]' % (100*np_probabilities[index], index))
        result_text+=str(sorted_inds[0])+'=>'+str( "{0:.2f}".format(100*np_probabilities[sorted_inds[0]]))+'%\n'
        print(np_images.shape)
        plt.figure()
        plt.imshow(np_images[0].astype(np.uint8))
        plt.text(225,225,result_text,horizontalalignment='center', verticalalignment='center',fontsize=21,color='blue')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    tf.app.run()
