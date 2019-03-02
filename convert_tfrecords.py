import tensorflow as tf
import config as cfg
import os
import xml.etree.ElementTree as ET
import random
import sys
import numpy as np
# sys.path.append('..')
slim = tf.contrib.slim



class dataTransform:
    def __init__(self, output_dir='./tfrecords',shuffle=True):
        self.is_data_exits = True
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.is_data_exits = False
        elif not os.path.exists(output_dir + "/voc_train_000.tfrecords"):
            self.is_data_exits = False
        self._output_dir = output_dir
        self._data_dir = cfg.DATA_DIR
        self._annotation_dir = os.path.join(self._data_dir,cfg.DIRECTORY_ANNOTATIONS)
        self._image_dir = os.path.join(self._data_dir,cfg.DIRECTORY_IMAGES)
        self._filenames = sorted(os.listdir(self._annotation_dir))
        self._example_per_file = cfg.EXAMPLE_PER_FIEL
        self._voc_label_ind = cfg.VOC_LABELS
        self._image_size = cfg.IMG_SIZE
        if shuffle:
            random.seed(2019)
            random.shuffle(self._filenames)

    def get(self):
        i = 0
        fidx = 0
        while i < len(self._filenames):
            tf_filename = self._output_dir + '/voc_train_{:03d}.tfrecords'.format(fidx)
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                j = 0
                while i < len(self._filenames) and j < self._example_per_file:
                    sys.stdout.write('\r>> Converting image %d/%d' % (i+1,len(self._filenames)))
                    sys.stdout.flush()

                    filename = self._filenames[i]
                    img_name = filename[:-4]
                    self._add_to_tfrecords(tfrecord_writer,img_name)
                    i += 1
                    j += 1
                fidx += 1
        print('\nFinished converting the Pascal VOC dataset!')


    def _add_to_tfrecords(self,tfrecord_writer,img_name):
        image_data,shape,bboxes,labels = self._process(img_name)
        example = self._convert_to_example(image_data,shape,bboxes,labels)
        tfrecord_writer.write(example.SerializeToString())
    def _process(self,img_name):
        filename = self._image_dir + img_name + '.jpg'
        image_data = tf.gfile.FastGFile(filename,'rb').read()
        filename = self._annotation_dir + img_name + '.xml'

        tree = ET.parse(filename)
        root = tree.getroot()

        size = root.find('size')
        shape = [int(size.find('height').text),
                 int(size.find('width').text),
                 int(size.find('depth').text)]

        bboxes = []
        labels = []
        h_ratio = 1.0 * self._image_size / shape[0]
        w_ratio = 1.0 * self._image_size / shape[1]

        for obj in root.findall('object'):
            label = obj.find('name').text
            labels.append(int(self._voc_label_ind[label][0]))

            bbox = obj.find('bndbox')
            # 0 indexes and clip
            # 这里保存448图像大小下的真实box
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio,self._image_size),0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio,self._image_size),0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio,self._image_size),0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio,self._image_size),0)

            bboxes.append((x1,y1,x2,y2))
        return image_data, shape,bboxes,labels

    def _convert_to_example(self,image_data,shape,bboxes,labels):
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        for b in bboxes:
            assert len(b) == 4
            [l.append(point) for l,point in zip([xmin,ymin,xmax,ymax],b)]
        image_format = b'JPEG'
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
        return example

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


if __name__ == '__main__':
    a = dataTransform()
    a.get()