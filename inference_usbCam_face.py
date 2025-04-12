# 本文件部分代码基于yeephycho的tensorflow-face-detection项目（https://github.com/yeephycho/tensorflow-face-detection），遵循Apache 2.0许可。
# 请参考项目README.md或LICENSE文件获取完整许可信息。


#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2

from utils import label_map_util
from utils import visualization_utils_color as vis_util

# sys：用于处理命令行参数。
# time：用于计算推理时间。
# numpy：用于处理图像数组。
# tensorflow：用于加载和运行预训练的人脸检测模型。
# cv2：OpenCV库，用于处理视频流和图像。
# label_map_util和visualization_utils_color：自定义工具函数，用于处理标签映射和可视化检测结果。


# 预训练的冻结推理图的路径。
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# 标签映射文件的路径。
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

# 检测的类别数量。
NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# load_labelmap：从文本文件中加载标签映射。
# convert_label_map_to_categories：将标签映射转换为适合评估的类别列表。
# create_category_index：创建一个以类别ID为键的类别索引字典。


# _init_方法：
#     创建预训练的冻结推理图
#     创建一个TensorFlow会话，并配置GPU内存使用。
# run方法：
#     将输入图像从BGR转换为RGB格式。
#     扩展图像维度以适应模型输入。
#     获取模型的输入和输出张量。
#     运行模型进行推理，并记录推理时间。
#     返回检测到的边界框、得分、类别和检测数量。


class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True


    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)



# 检查命令行参数是否正确。
# 尝试将命令行参数转换为整数作为摄像头ID，如果失败则将其作为视频文件路径。
# 创建TensoflowFaceDector类的实例。
# 打开视频捕获设备：
#     进入循环，不断读取视频帧：翻转图像以实现镜像效果。
#     调用TensorflowFaceDector类的run方法进行人脸检测。
#     调用visualization_utils_color模块的visualize_boxes_and_labels_on_image_array函数在图像上绘制检测结果。
#     显示处理后的图像。
#     检查是否按下q或Esc键，如果是则退出循环。
# 释放视频捕获设备。

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print ("usage:%s (cameraID | filename) Detect faces\
 in the video example:%s 0"%(sys.argv[0], sys.argv[0]))
        exit(1)

    try:
    	camID = int(sys.argv[1])
    except:
    	camID = sys.argv[1]
    
    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    cap = cv2.VideoCapture(camID)
    windowNotSet = True
    while True:
        ret, image = cap.read()
        if ret == 0:
            break

        [h, w] = image.shape[:2]
        print (h, w)
        image = cv2.flip(image, 1)

        (boxes, scores, classes, num_detections) = tDetector.run(image)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4)

        if windowNotSet is True:
            cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
            windowNotSet = False

        cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
