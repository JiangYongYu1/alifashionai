import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os
import pandas as pd
import read_image_test
train_image_size = 299
CHANAL = 3
classnum = 8
total_step = 2147
slim = tf.contrib.slim

def softmax(prob):
    sum = 0
    for pr in prob:
        sum += pr
    prob = [str(round(pr/sum, 4)) for pr in prob]
    pprob = ';'.join(prob)
    return pprob

with tf.name_scope('input'):
    x_input = tf.placeholder(dtype=tf.float32, shape=[None, train_image_size, train_image_size, CHANAL])
with tf.name_scope('model'):
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(x_input, classnum, is_training=False)


with tf.name_scope('input_val_image'):
    test_file_path = os.path.join('F:/alicloth/z_rank/tfrecord/coat_length_labels/test_0.tfrecords')
    test_image_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(test_file_path))
    test_images, test_labels = read_image_test.read_test_tfrecord(test_image_filename_queue, 1)

saver = tf.train.Saver()
outdict = {}
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
    for i in range(total_step):
        testimages, testlabels = sess.run([test_images, test_labels])
        prob = sess.run(end_points['Predictions'], feed_dict={x_input: testimages})
        testima = bytes.decode(testlabels[0], encoding='utf8')
        probs = softmax(prob[0])
        iname = '/'.join(testima.split('/')[-3:])
        outdict[iname] = probs
    dataout = pd.Series(outdict)
    dataout = pd.DataFrame(dataout)
    dataout['name'] = dataout.index
    dataout['id'] = list(range(total_step))
    dataout.set_index('id', inplace=True)
    dataout.to_csv('F:/alicloth/z_rank/out/coat.csv', index=None)
    coord.request_stop()
    coord.join(threads)