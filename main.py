import tensorflow as tf
import numpy as np
from precessing import precess
from convert_tfrecords import dataTransform
from get_blob import get_blob
from tools.yolo_net import YOLONet
import gc
from test import Detector
import config as cfg
slim = tf.contrib.slim
import matplotlib.pyplot as plt
import cv2

def main():
    dataGeter = dataTransform()
    if dataGeter.is_data_exits == False:
        dataGeter.get()
    del dataGeter;gc.collect()

    image_test = cv2.imread('./test/2.jpg')
    # image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
    # cv2.imshow('iamge', image_test)
    # cv2.waitKey(0)
    image_test_net = image_test.astype(np.float32)
    image_test_net = cv2.resize(image_test_net,(448,448))
    image_test_net = (image_test_net / 255.0) * 2.0 - 1.0
    image_test_net = np.expand_dims(image_test_net,0)

    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        with tf.device("/cpu:0"):
            global_step = tf.train.create_global_step()
            blob = get_blob()
            # (H,W,3) (3,) (?,) (?,4)(xmin,ymin,xmax,ymax)已经转换到448下的bboxes
            image,shape,labels,bboxes = blob.get()

            precess_gt = precess()
            image_data,gt = precess_gt.run(image,shape,labels,bboxes)

            r = tf.train.batch([image_data,gt],batch_size=cfg.BATCH_SIZE,
                               num_threads=4,
                               capacity=5 * cfg.BATCH_SIZE)
            batch_queue = slim.prefetch_queue.prefetch_queue(r,capacity=2)

            learning_rate = tf.train.exponential_decay(0.0001,global_step,
                                                       30000,0.1,True,name='learning_rate')
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)


        b_image,b_gt = batch_queue.dequeue()
        net = YOLONet()
        net_test = YOLONet(is_training=False)

        logits = net.build_network(b_image, num_outputs=net.output_size, alpha=net.alpha)

        logits_test = net_test.build_network(image_test_net, num_outputs=net.output_size, alpha=net.alpha,reuse=True)

        # cv2.imshow('iamge', image_test)
        # cv2.waitKey(0)

        is_that_success = tf.py_func(net.forward_test,[image_test,logits_test,tf.train.get_global_step()],tf.bool)

        net.loss_layer(logits, b_gt)
        total_loss = tf.losses.get_total_loss()

        variables_to_restore = slim.get_variables_to_restore(exclude=['global_step'])

        variables_to_train = tf.trainable_variables()
        train_op = optimizer.minimize(total_loss,var_list=variables_to_train)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config = tf.ConfigProto(gpu_options=gpu_options)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        saver = tf.train.Saver(var_list=variables_to_restore,max_to_keep=10,
                               write_version=2)

        if True:
            import time

            model_path = cfg.MODEL_DIR

            with tf.Session(config=config) as sess:

                summary = tf.summary.merge_all()
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners()
                writer = tf.summary.FileWriter(model_path,sess.graph)
                sess.run(init_op)

                saver.restore(sess,'./pretrained/YOLO.ckpt')
                for step in range(20000):
                    start_time = time.time()

                    _,loss_value = sess.run([train_op,total_loss])
                    duration = time.time() - start_time

                    if step % 10 == 0:
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str,step)
                        examples_per_sec = cfg.BATCH_SIZE / duration
                        sec_per_batch = float(duration)
                        format_str = "[*] step %d,  loss=%.2f (%.1f examples/sec; %.3f sec/batch)"
                        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
                    if step % 1000 == 0:
                        saver.save(sess, model_path+"yolov1.model", global_step=tf.train.get_global_step())
                    if step % 500 == 0:
                        cv2.imshow('iamge', image_test)
                        cv2.waitKey(0)
                        sess.run(is_that_success)

                coord.request_stop()
                coord.join(threads)

        # sess = tf.InteractiveSession()
        # tf.train.start_queue_runners(sess=sess)
        # a,b = sess.run([b_image,b_gt])
        # print("<><><><><>")




# 测试代码
# sess = tf.InteractiveSession()
# tf.train.start_queue_runners(sess=sess)
# a,b,c,d = sess.run([image,labels,shape,bboxes])
# a1,b1,c1,d1 = sess.run([image,labels,shape,bboxes])


if __name__ == '__main__':
    main()