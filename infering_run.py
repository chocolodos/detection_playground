import os
import cv2
import sys
import time
import argparse
import multiprocessing
# todo: socket import

from PIL import Image
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


# This is needed since the notebook is stored in the object_detection folder
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util
# Socket import
from utils import python_socket3 as pysock



# Get checkpoint map
CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for
# the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'Tensorflow_detection_model_zoo',
                            MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt')


# Label map(map indices to category names)
NUM_CLASSES  = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# HELPER CODE -> convert image
def load_image_into_numpy_array(image):
    '''Output: up.array of image'''
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape( (im_height, im_width, 3)).astype(np.uint8)
def save_nparray_into_img(array, name = 'saved_boxed_file.jpeg'):
    im = Image.fromarray(array)
    SAVE_PATH = os.path.join(CWD_PATH, 'test_images')
    im.save("test_images/saved_boxed_file.jpeg")
    print('Image saved to {}'.format(SAVE_PATH))

# 出 x_min,y_min, x_max, y_max, 和类（eg：person）
def box_N_class_output(boxes,
                       classes,
                       scores,
                       category_index,
                       max_boxes_to_return=20,
                       min_score_thresh=.5):
    class_map = []
    box_map = []
    score_map = []
    if not max_boxes_to_return:
        max_boxes_to_return = boxes.shape[0]
    for i in range(min(max_boxes_to_return, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = boxes[i].tolist()
            box_map.append(box)
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]['name']
            else:
                class_name = 'N/A'

            class_map.append(class_name)
            score_map.append(scores[i])
    num_detected = len(score_map)
    return box_map, score_map, class_map, num_detected

# DETECTION
def detect_objects(image_np, sess, detection_graph):
    ''' Input: Original_test_image,
               tf.Session(),
               the_frozen_TF_graph
        Output: Labelled_and_Boxed_image '''

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0) #[None,None,3] -> [1,None,None,3]

    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    # Selecting high possibility results
    (boxes, scores, classes, num_detections) = box_N_class_output(
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            max_boxes_to_return=20)


    return image_np, boxes, scores, classes, num_detections

# Load frozen TF model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()

    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Image PATH
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,
        'image{}.jpg'.format(i)) for i in range(0, 1) ]

# Size, in inches, of the output images. ->(how large the result plt is)
IMAGE_SIZE = (12, 8)


# INFERING
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)

            image_boxed, boxes, scores, classes, num_detection = detect_objects(
                            image_np, sess, detection_graph)
            print(scores,classes)

save_nparray_into_img(image_boxed)

# Combine boxes with classes
for i in range(len(classes)):
    boxes[i].append(classes[i])

import csv
with open("boxes_with_classes.csv", "w") as f:
    w = csv.writer(f)
    w.writerows(boxes)

print("=============Finished boxing===============")

isSending = False

if isSending:
    # Sending message through
    s = pysock.Setup()
    bytes_box =  str.encode(str(boxes))
    s.sendall(bytes_box)
    s.close()





print("=============Finished sending===============")
