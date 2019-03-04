import tensorflow as tf
import numpy as np
import config as cfg
slim = tf.contrib.slim
from tools.yolo_net import YOLONet
import cv2
import time

class Detector(object):
    def __init__(self,net,sess=None):
        self.net = net
        self.num_class = cfg.CLASSES
        self.cell_size = cfg.CELL_SIZE
        self.image_size = cfg.IMG_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell
        self._classa = cfg.CLASSES_a

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def image_detector(self,imname,step,weight_file,wait=0):
        self.saver.restore(self.sess,weight_file)
        image = cv2.imread(imname)
        start_time = time.time()
        result = self.detect(image)
        duration = time.time() - start_time
        self.draw_result(image,result)
        print("Average detecting time:  {:.3f}s".format(duration))
        cv2.imwrite("./test/{}.jpg".format(step),image)
        # cv2.imshow('Image',image)
        # cv2.waitKey(wait)

    def interpret_output(self, output):
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))
        class_probs = np.reshape(
            output[0:self.boundary1],
            (self.cell_size, self.cell_size, self.num_class))
        scales = np.reshape(
            output[self.boundary1:self.boundary2],
            (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(
            output[self.boundary2:],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell)
        offset = np.transpose(
            np.reshape(
                offset,
                [self.boxes_per_cell, self.cell_size, self.cell_size]),
            (1, 2, 0))
        # x的offset
        boxes[:, :, :, 0] += offset
        # y的offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        # 除以cell_size获取相对img_size的位置
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        # 平方w h
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])
        # 换算到448图大小拿到448分辨率下的的boxes
        boxes *= self.image_size
        # 按论文公式计算分类置信度
        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])
        # probs(7,7,2,20)
        # 大于设定阈值的置信度才能进行后续计算
        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        # 拿到上面大于阈值的tensor的非零值的位置
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(
            filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [self._classa[classes_num_filtered[i]],
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])

        return result


    def detect_from_cvmat(self,inputs):
        net_output = self.sess.run(self.net.logits,feed_dict={self.net.images:inputs})
        results = []
        # batchsize
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))
        return results

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def detect(self,image):
        img_h, img_w, _ = image.shape
        inputs = cv2.resize(image, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)

        return result



    def draw_result(self,image,result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3]/2)
            h = int(result[i][4]/2)
            cv2.rectangle(image, (x - w,y - h),(x + w,y + h),(0,255,0),2)
            cv2.rectangle(image, (x - w,y - h -20),
                          (x + w,y - h),(125,125,125),-1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(image,
                        result[i][0] + ' : %.2f' % result[i][5],
                        (x - w + 5,y - h - 7),cv2.FONT_HERSHEY_SIMPLEX,0.5,
                        (0,0,0),1,lineType)

def main():

    yolo = YOLONet(False)
    detector = Detector(yolo)

    image_name = './test/person.jpg'
    detector.image_detector(image_name,0)

if __name__ == '__main__':
    main()