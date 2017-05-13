import tensorflow as tf
from PIL import Image

filenames = ['/home/cideep/Work/tensorflow/datasets/VOC-2012/VOC-2012-train/JPEGImages/2007_000032.jpg']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=1)
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
image = tf.image.decode_jpeg(value)

# PROBLEM HERE!
resized_image = tf.image.resize_image_with_crop_or_pad(image, 200, 200)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess = tf.Session()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

img = sess.run(resized_image)
print('image shape', img.shape)
img = Image.fromarray(img, "RGB")
img.show('image')
