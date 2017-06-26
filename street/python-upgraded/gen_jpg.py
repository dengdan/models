from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import parsing_ops

import numpy as np
import skimage.io as io
import util

data_files = [util.io.get_absolute_path('~/dataset_nfs/FSNS/train-00001-of-00512')]
filename_queue = tf.train.string_input_producer(
      data_files, capacity=100)
reader = tf.TFRecordReader()
_, example_serialized = reader.read(filename_queue)
example_serialized = tf.reshape(example_serialized, shape=[])
features = tf.parse_single_example(
  example_serialized,
  {'image/encoded': parsing_ops.FixedLenFeature(
      [1], dtype=tf.string, default_value=''),
   'image/text': parsing_ops.FixedLenFeature(
       [1], dtype=tf.string, default_value=''),
   'image/class': parsing_ops.VarLenFeature(dtype=tf.int64),
   'image/unpadded_class': parsing_ops.VarLenFeature(dtype=tf.int64),
   'image/height': parsing_ops.FixedLenFeature(
       [1], dtype=tf.int64, default_value=1),
   'image/width': parsing_ops.FixedLenFeature(
       [1], dtype=tf.int64, default_value=1)})
#labels = features['image/unpadded_class']
labels = features['image/class']
labels = tf.serialize_sparse(labels)
image_buffer = tf.reshape(features['image/encoded'], shape=[], name='encoded')
image = tf.image.decode_png(image_buffer, channels=3)

height = tf.reshape(features['image/height'], [-1])
width = tf.reshape(features['image/width'], [-1])
text = tf.reshape(features['image/text'], shape=[])

images_and_label_lists = [[image, height, width, labels, text]]

images, heights, widths, labels, texts = tf.train.batch_join(
      images_and_label_lists,
      batch_size=1,
      capacity=16 * 10,
      dynamic_pad=True)
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)  
    data = []
    for i in range(100):
        imgs, txts= sess.run([images, texts])
        #util.img.imwrite('~/temp_nfs/no-use/fsns/%d.jpg'%(i), imgs[0, ...], rgb = True)
        image_data = imgs[0, ...]
        image_data = np.transpose(image_data, [0, 2, 1]);
        image_data = np.reshape(image_data, [150, 3, 4, 150])
        image_data = np.transpose(image_data, [2, 0, 3, 1])
        for idx, I in enumerate(image_data):
#            util.cit(I)
            util.img.imwrite('~/temp_nfs/no-use/fsns/%d_%d.jpg'%(i, idx), I, rgb = True)
    coord.request_stop()  
    coord.join(threads)
