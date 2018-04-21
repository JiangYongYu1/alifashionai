import pandas as pd
import os
import tensorflow as tf
import sys



def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def image_to_tfexample(image_data, image_format, height, width, imagename):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/name': bytes_feature(imagename),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))

def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = '%s_%d.tfrecords' % (
      split_name, shard_id)
  return os.path.join(dataset_dir, output_filename)

def train_input(path):
    with open(path, 'r') as f:
        Images = f.readlines()
    Images = [Image.split('\n')[0] for Image in Images]
    frontpath = '/'.join(path.split('/')[0: 5])
    backpath = path.split('/')[-2]
    pathlabels = os.path.join(os.path.join(frontpath, 'annos'), backpath + '.csv')
    print(pathlabels)
    la = pd.read_csv(pathlabels)
    labels = []
    for imagename in Images:
        ima = '/'.join(imagename.split('/')[4:])
        label = la.loc[list(la['name']).index(ima), 'getlabel']
        label = label[1: -1].split(',')
        label = [float(i) for i in label]
        labels.append(label)
    return Images, labels

def _convert_dataset(split_name, filenames, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['test']

  _NUM_SHARDS = 1

  with tf.Graph().as_default():
    image_reader = ImageReader()
    dataset_dirs = dataset_dir.split('/')
    print(dataset_dirs)
    dataset_dir = '/'.join(dataset_dirs[0: -2])
    dataset_dir = dataset_dir + '/tfrecord/'
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    dataset_dir = dataset_dir + dataset_dirs[-1]
    print(dataset_dir)
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          for i in range(len(filenames)):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)
            example = image_to_tfexample(
                image_data, b'jpg', height, width, bytes(filenames[i], encoding='utf8'))
            tfrecord_writer.write(example.SerializeToString())


def run(data_dir):

  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """

  test_data_dir = os.listdir(data_dir)

  test_data_filename = [data_dir + '/' + imname for imname in test_data_dir]

  # First, convert the training and validation sets.
  _convert_dataset('test', test_data_filename,
                   data_dir)


  print('\nFinished converting the Fashion dataset!')







data_dir = 'F:/alicloth/z_rank/Images'

label_count = {
    'coat_length_labels': 8,
    'collar_design_labels': 5,
    'neck_design_labels': 5,
    'neckline_design_labels': 10,
    'pant_length_labels': 6,
    'skirt_length_labels': 6,
    'sleeve_length_labels': 9,
    'lapel_design_labels': 5
}
for key in label_count.keys():
    image_data_dir = data_dir + '/' + key
    run(image_data_dir)