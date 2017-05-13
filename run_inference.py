import numpy as np
import os
import tensorflow as tf
# import urllib2
from urllib.request import urlopen
import matplotlib.pyplot as plt

from nets import inception
from preprocessing import inception_preprocessing

slim = tf.contrib.slim
image_size = inception.inception_v3.default_image_size

tf.app.flags.DEFINE_string(
    'model_name', 'inception_resnet_v2', 'Model name to use')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/home/cideep/Work/tensorflow/checkpoints/my-fine-tuned/inc-resnet-v2-cifar10/model.ckpt-50000',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_integer(
    'num_classes', 10,
    'Number of output classes.')

FLAGS = tf.app.flags.FLAGS

name_to_model = {'inception_v4': inception.inception_v4,
                 'inception_resnet_v2': inception.inception_resnet_v2}
name_to_arg_scope = {'inception_v4': inception.inception_v4_arg_scope,
                     'inception_resnet_v2': inception.inception_resnet_v2_arg_scope}

def main(_):
    with tf.Graph().as_default():

        url = 'http://www.slate.com/content/dam/slate/articles/technology/future_tense/2016/05/160503_FT_cybersecurity-airplanes.jpg.CROP.promo-xlarge2.jpg'
        image_string = urlopen(url).read()
        image = tf.image.decode_jpeg(image_string, channels=3)
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images  = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        net_logit = name_to_model[FLAGS.model_name]
        arg_scope = name_to_arg_scope[FLAGS.model_name]
        with slim.arg_scope(arg_scope()):
            logits, _ = net_logit(processed_images, num_classes=FLAGS.num_classes,
                                  is_training=False)

        probabilities = tf.nn.softmax(logits)
        init_fn = slim.assign_from_checkpoint_fn(
            FLAGS.checkpoint_path,
            slim.get_model_variables('InceptionResnetV2'))

        with tf.Session() as sess:
            init_fn(sess)
            np_image, probabilities = sess.run([image, probabilities])
            probabilities = probabilities[0, 0:]
            print('probabilities:', probabilities)
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
        # names = imagenet.create_readable_names_for_imagenet_labels()
        result_text=''
        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.2f%% => [%d]' % (100*probabilities[index], index))
        result_text+=str(sorted_inds[0])+'=>'+str( "{0:.2f}".format(100*probabilities[sorted_inds[0]]))+'%\n'
        plt.figure()
        plt.imshow(np_image.astype(np.uint8))
        plt.text(225,225,result_text,horizontalalignment='center', verticalalignment='center',fontsize=21,color='blue')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
  tf.app.run()
