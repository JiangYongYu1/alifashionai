import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os
import read_image
train_image_size = 299
CHANAL = 3
classnum = 8

slim = tf.contrib.slim
with tf.name_scope('input'):
    x_input = tf.placeholder(dtype=tf.float32, shape=[None, train_image_size, train_image_size, CHANAL])
    y_input = tf.placeholder(dtype=tf.float32, shape=[None, classnum])
with tf.name_scope('model'):
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(x_input, classnum, is_training=False)
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y_input, 1), tf.arg_max(end_points['Predictions'], 1)), tf.float32))


with tf.name_scope('input_val_image'):
    test_file_path = os.path.join('F:/alicloth/base/trceshi/coat_length_labels/validation/validation_00000.tfrecords')
    test_image_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(test_file_path))
    test_images, test_labels = read_image.read_val_tfrecord(test_image_filename_queue, classnum, 10)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    ckpt = tf.train.get_checkpoint_state('F:/alicloth/base/trained_model/coat')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restored %s" % ckpt.model_checkpoint_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    acc_mun = 0
    ia = 0
    for i in range(int(1133/10)):
        testimages, testlabels = sess.run([test_images, test_labels])
        acu, pro = sess.run([accuracy, end_points['Predictions']], feed_dict={x_input: testimages, y_input: testlabels})
        acc_mun += acu
        print(acu)
        print(pro)
        ia += 1
    print(acc_mun/ia)
    coord.request_stop()
    coord.join(threads)