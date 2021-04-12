import itertools
import os
import tempfile
import urllib
from pprint import pprint
from time import sleep

import cv2
import imutils

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

os.environ["TFHUB_CACHE_DIR"] = '/tmp/tfhub'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['PYTHONPATH'] += "models"
print(os.path.join(tempfile.gettempdir(), "tfhub_modules"))

print("TF version:", tf.__version__)
# print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

print("loading model")

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = "labels.txt"
NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# pipeline_config = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config'
# model_dir = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint'
#
# # Load pipeline config and build a detection model
# configs = config_util.get_configs_from_pipeline_file(pipeline_config)
# model_config = configs['model']
# model = model_builder.build(
#     model_config=model_config, is_training=False)
#
# # Restore checkpoint
# ckpt = tf.compat.v2.train.Checkpoint(model=model)
# ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

label_map_path = 'labels.txt'
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

print("done")

cap = cv2.VideoCapture(700)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)

dimensions = 640

print("warming camera")
sleep(5)

print("starting loop")

brightness = 60
#
# while True:
#     ret, frame = cap.read()
#
#     if cv2.waitKey(1) == ord('w'):
#         brightness += 5
#         cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
#     elif cv2.waitKey(1) == ord('s'):
#         brightness -= 5
#         cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
#     elif cv2.waitKey(1) == ord('z'):
#         break
#
#     cv2.imshow("frame", frame)



def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(io.BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.resize(frame, (1138, 640))

    # image_path = "imagetest2.jpg"
    # image_np = load_image_into_numpy_array(image_path)
    #
    # frame = image_np
    cv2.imshow("frame", frame)

    frame = frame[int((len(frame) - dimensions)/2):int(len(frame) - (len(frame) - dimensions)/2), int((len(frame[0]) - dimensions)/2):int(len(frame[0]) - (len(frame[0]) - dimensions)/2)]

    img = frame

    input_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    input_tensor = input_tensor[tf.newaxis,...]

    image, shapes = model.preprocess(input_tensor)


    prediction_dict = model.predict(input_tensor, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    shapes = tf.reshape(shapes, [-1])

    image_np_with_detections = tf.cast(tf.squeeze(input_tensor), dtype=tf.uint8).numpy()

    saved = image_np_with_detections

    # cv2.imshow("preplot", image_np_with_detections)
    # sleep(2)

    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + 1).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=10,
            min_score_thresh=.30,
            agnostic_mode=False)


    # plt.figure(figsize=(12,16))
    cv2.imshow("plotted", image_np_with_detections)
    # plt.show()

    # plt.imshow(frame)
    # plt.show()
    # cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
