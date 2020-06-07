import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import cv2
import time
from collections import defaultdict
sys.path.append("/home/pi/models/research/")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
MODEL_NAME = '/home/pi/models/research/object_detection/models/ssd_mobilenet_v1_coco_2018_01_28'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('/home/pi/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
model_path = "/home/pi/models/research/object_detection/models/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt"
start = time.clock()
NUM_CLASSES = 90
end= time.clock()
print('load the model' ,(end -start))
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
cap = cv2.VideoCapture(0)
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(tf.global_variables_initializer())
        loader = tf.train.import_meta_graph(model_path + '.meta')
        loader.restore(sess, model_path)
        while(1):
            start = time.clock()
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            image_np =frame
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np, np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=6)
            end = time.clock()
            print 'One frame detect take time:' ,end - start
            cv2.imshow("capture", image_np)
            print('after cv2 show')
            cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
