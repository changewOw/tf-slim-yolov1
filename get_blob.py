import tensorflow as tf
import numpy as np
import os
import config

slim = tf.contrib.slim

class get_blob:
    def __init__(self,tfrecords_dir='./tfrecords'):
        self._tfrecords_dir = tfrecords_dir
        self._reader = tf.TFRecordReader

        self.dataset = self._get_split()
        self.provider = self._get_provider()
    def _get_split(self):
        # 这里获取数据
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/width': tf.FixedLenFeature([1], tf.int64),
            'image/channels': tf.FixedLenFeature([1], tf.int64),
            'image/shape': tf.FixedLenFeature([3], tf.int64),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64)
        }
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
            'shape': slim.tfexample_decoder.Tensor('image/shape'),
            'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['xmin', 'ymin', 'xmax', 'ymax'], 'image/object/bbox/'),
            'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        }
        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)
        items_to_descriptions = {
            'image': 'A color image of varying height and width.',
            'shape': 'Shape of the image',
            'object/bbox': 'A list of bounding boxes, one per each object.',
            'object/label': 'A list of labels, one per each object.',
        }
        return slim.dataset.Dataset(
                    data_sources='./tfrecords/voc_train_*.tfrecords',
                    reader=self._reader,
                    decoder=decoder,
                    num_samples=400, # 只用2007voc 400张图片训练，过拟合训练集看效果
                    items_to_descriptions=items_to_descriptions,
                    num_classes=20,
                    labels_to_names=None)

    def _get_provider(self):
         return slim.dataset_data_provider.DatasetDataProvider(
            self.dataset,
            num_readers=4,
            common_queue_capacity=20 * 16,
            common_queue_min=10 * 16,
            shuffle=False)
    def get(self):
        return self.provider.get(['image','shape',
                                  'object/label',
                                  'object/bbox'])
