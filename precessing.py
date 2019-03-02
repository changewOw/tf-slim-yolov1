import numpy as np
import tensorflow as tf
import config as cfg
class precess:
    def __init__(self):
        self._img_size = cfg.IMG_SIZE
        self._cell_size = cfg.CELL_SIZE

        self._gt = tf.zeros((self._cell_size,self._cell_size,25))


    def run(self,image_data,shape,labels,bboxes):
        image_data = self._image_precess(image_data)
        self._bboxes_precess(shape,labels,bboxes)
        return image_data,self._gt

    def _image_precess(self,image_data):
        image_data = tf.image.resize_images(image_data,(self._img_size,self._img_size))
        image_data = tf.image.convert_image_dtype(image_data,dtype=tf.float32)
        image_data = (image_data / 255.0) * 2.0 - 1.0
        return image_data

    def _bboxes_precess(self,shape,labels,bboxes):
        i = 0
        [i,labels,bboxes,self._gt] = tf.while_loop(self._condition,self._body,
                                             [i,labels,bboxes,self._gt])

    def _condition(self,i,labels,bboxes,gt):
        r = tf.less(i,tf.shape(labels))
        return r[0]

    def _body(self,i,labels,bboxes,gt):
        cls_ind = labels[i]
        # x1 y1 x2 y2 = bboxes[i]
        boxes = [(bboxes[i,2]+bboxes[i,0])/2.0,(bboxes[i,3]+bboxes[i,1])/2.0,bboxes[i,2]-bboxes[i,0],bboxes[i,3]-bboxes[i,1]]
        x_ind = tf.cast(boxes[0]*self._cell_size / self._img_size,tf.int32)
        y_ind = tf.cast(boxes[1]*self._cell_size / self._img_size,tf.int32)

        gt = tf.py_func(imple_np,[gt,y_ind,x_ind,cls_ind,boxes],tf.float32)
        gt.set_shape([7,7,25])
        return [i+1,labels,bboxes,gt]

        # if tf.equal(gt[y_ind,x_ind,0],1):
        #     return [i+1,labels,bboxes,gt]
        # gt[y_ind,x_ind,0] = 1
        # gt[y_ind,x_ind,1:5] = boxes
        # gt[y_ind,x_ind,5+cls_ind] = 1
        # return [i+1,labels,bboxes,gt]
def imple_np(gt,y_ind,x_ind,cls_ind,boxes):
    if gt[y_ind,x_ind,0] == 1:
        return gt
    gt[y_ind,x_ind,0] = 1
    gt[y_ind,x_ind,1:5] = boxes
    gt[y_ind,x_ind,5 + cls_ind] = 1
    return gt

