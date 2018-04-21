import tensorflow as tf
import inception_preprocessing as processing
slim = tf.contrib.slim
#VGG_MEAN = [103.939, 116.779, 123.68]

def read_test_tfrecord(filename_queue, batch_size):
    feature = {
        'image/encoded': tf.FixedLenFeature((), tf.string),
        'image/format': tf.FixedLenFeature((), tf.string),
        'image/name': tf.FixedLenFeature([], tf.string),
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
    imagename = features['image/name']

    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    # restore image to [height, width, 3]
    image = tf.reshape(image, [height, width, 3])
    # resize

    image = processing.preprocess_image(image, 299, 299, is_training=False)

    images, imagenames = tf.train.batch([image, imagename], batch_size=batch_size, capacity=3*batch_size, num_threads=10)
    return images, imagenames
