import tensorflow as tf
import inception_preprocessing as processing
slim = tf.contrib.slim
#VGG_MEAN = [103.939, 116.779, 123.68]
def read_train_tfrecord(filename_queue, classnum, batch_size):
    feature = {
        'image/encoded': tf.FixedLenFeature((), tf.string),
        'image/format': tf.FixedLenFeature((), tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
    }

    reader = tf.TFRecordReader()
    # read in serialized example data
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.image.decode_jpeg(features['image/encoded'])
    image = tf.image.convert_image_dtype(image,
                                         dtype=tf.float32)  # convert dtype from unit8 to float32 for later resize
    label = tf.cast(features['image/class/label'], tf.int64)

    label = slim.one_hot_encoding(
        label, classnum)
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    # restore image to [height, width, 3]
    image = tf.reshape(image, [height, width, 3])
    # resize
    image = tf.image.resize_images(image, [304, 304])

    image = processing.preprocess_image(image, 299, 299, is_training=True)
    # red, green, blue = tf.split(image, 3, 2)
    #image = tf.concat([
    #        blue - VGG_MEAN[0],
    #        green - VGG_MEAN[1],
    #        red - VGG_MEAN[2]], 2)
    # create bathch
    #print(image)
    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=3*batch_size, num_threads=10,
                                            min_after_dequeue=10)
    return images, labels


def read_val_tfrecord(filename_queue, classnum, batch_size):
    feature = {
        'image/encoded': tf.FixedLenFeature((), tf.string),
        'image/format': tf.FixedLenFeature((), tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
    }

    reader = tf.TFRecordReader()
    # read in serialized example data
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.image.decode_jpeg(features['image/encoded'])
    image = tf.image.convert_image_dtype(image,
                                         dtype=tf.float32)  # convert dtype from unit8 to float32 for later resize
    label = tf.cast(features['image/class/label'], tf.int64)

    label = slim.one_hot_encoding(
        label, classnum)
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    # restore image to [height, width, 3]
    image = tf.reshape(image, [height, width, 3])
    # resize

    image = processing.preprocess_image(image, 299, 299, is_training=False)
    # red, green, blue = tf.split(image, 3, 2)
    #image = tf.concat([
    #        blue - VGG_MEAN[0],
    #        green - VGG_MEAN[1],
    #        red - VGG_MEAN[2]], 2)
    # create bathch
    #print(image)
    images, labels = tf.train.batch([image, label], batch_size=batch_size, capacity=3*batch_size, num_threads=10)
    return images, labels
