import os
import pathlib
import sys
import tarfile

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import cv2
from mss import mss

import tensorflow as tf

sys.path.append("models/research")
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import ops as utils_ops


BASE_DIR = "E:/Projects/fps/"
MODELS_DIR = os.path.join(BASE_DIR, "workspace/models/")
# MODEL_NAME = 'centernet_hg104_512x512_coco17_tpu-8'
# MODEL_CKPT = 'ckpt-0'
# MODEL_NAME = 'centernet_mobilenetv2fpn_512x512_coco17_od'
# MODEL_CKPT = 'ckpt-301'
# MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
# MODEL_CKPT = 'ckpt-0'
MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
MODEL_CKPT = 'ckpt-9'

# MODEL_VERSION = 'v1'
MODEL_VERSION = 'v3'

TEST_IMAGE_PATH = None  # os.path.join(BASE_DIR, "models/research/object_detection/test_images/image2.jpg")

SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
WINDOW_X, WINDOW_Y = 400, 225
OUTPUT_WINDOW_NAME = 'Object detection'


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
    im = Image.open(BytesIO(img_data))
    (im_width, im_height) = im.size
    return np.array(im.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_keypoint_tuples(eval_config):
    """Return a tuple list of keypoint edges from the eval config.

    Args:
    eval_config: an eval config containing the keypoint edges

    Returns:
    a list of edge tuples, each in the format (start, end)
    """
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list


def load_model(model_name=MODEL_NAME, model_ckpt=MODEL_CKPT, model_version=MODEL_VERSION, models_dir=MODELS_DIR):
    # Load pipeline config and build a detection model
    model_dir = os.path.join(models_dir, model_name, model_version)
    pipeline_config = os.path.join(model_dir, 'pipeline.config')
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model = model_builder.build(model_config=model_config, is_training=False)
    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=model)
    ckpt.restore(os.path.join(model_dir, model_ckpt)).expect_partial()
    return model, configs


def load_labels(eval_input_config):
    label_map_path = eval_input_config.label_map_path
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    ci = label_map_util.create_category_index(categories)
    return ci


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections

    return detect_fn


def run_inference_for_single_image(detection_func, image_np, cat_index, class_id=1, label_id_offset=1, eval_config=None):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    # Actual detection
    detections = detection_func(input_tensor)
    image_out = image_np.copy()
    boxes = []
    classes = []
    scores = []
    for i, x in enumerate(detections['detection_classes'][0]):
        if x + label_id_offset == class_id and detections['detection_scores'][0, i] > 0.5:
            boxes.append(detections['detection_boxes'][0, i])
            classes.append(int(x + label_id_offset))
            scores.append(detections['detection_scores'][0, i])
    boxes = np.array(boxes)
    classes = np.array(classes)
    scores = np.array(scores)
    # Use keypoints if available in detections
    if 'detection_keypoints' in detections and eval_config is not None:
        keypoints = detections['detection_keypoints'][0].numpy()
        keypoint_scores = detections['detection_keypoint_scores'][0].numpy()
        keypoint_kwargs = dict(
            keypoints=keypoints,
            keypoint_scores=keypoint_scores,
            keypoint_edges=get_keypoint_tuples(eval_config),
        )
    else:
        keypoint_kwargs = dict()
    # Visualization of the results of a detection
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_out,
        boxes,
        classes,
        scores,
        cat_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
        **keypoint_kwargs
    )
    return detections, image_out


def real_time_object_detection(window_name, window_shape, detection_func, image, cat_index, **kwargs):
    # Make sure image is a numpy array
    image = np.array(image)
    # Create labeled image using object detection
    detections, labeled_image = run_inference_for_single_image(detection_func, image, cat_index, **kwargs)
    cv2.imshow(window_name, cv2.resize(labeled_image, window_shape))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return None, None
    return detections, labeled_image


def create_output_window(window_name, window_pos):
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, window_pos[0], window_pos[1])
    return


if __name__ == "__main__":
    detection_model, detection_configs = load_model()
    get_detections = get_model_detection_function(detection_model)

    # Load label map (for plotting)
    category_index = load_labels(detection_configs['eval_input_config'])

    # Object detection on a single test image
    if TEST_IMAGE_PATH:
        test_image_np = load_image_into_numpy_array(TEST_IMAGE_PATH)
        labeled_test_image = run_inference_for_single_image(get_detections, test_image_np)
        plt.figure(figsize=(12, 16))
        plt.imshow(labeled_test_image)
        plt.savefig('test.png')

    # Output window for real-time object detection
    input_window = dict(
        left=SCREEN_WIDTH // 4 - WINDOW_X // 2,
        top=SCREEN_HEIGHT // 4 - WINDOW_Y // 2,
        width=WINDOW_X,
        height=WINDOW_Y,
    )
    output_pos = (SCREEN_WIDTH - WINDOW_X, SCREEN_HEIGHT - WINDOW_Y - 30)
    output_shape = (WINDOW_X, WINDOW_Y)
    create_output_window(OUTPUT_WINDOW_NAME, output_pos)

    # Real-time O.D.: webcam input
    cap = cv2.VideoCapture(0)
    while True:
        # Webcam capture
        ret, cap_arr = cap.read()
        output, _ = real_time_object_detection(
            OUTPUT_WINDOW_NAME,
            output_shape,
            get_detections,
            cap_arr,
            category_index,
            eval_config=detection_configs['eval_config'],
        )
        if output is None:
            break
    cap.release()

    # Real-time O.D.: monitor input
    with mss() as sct:
        while True:
            screenshot = sct.grab(input_window)
            screenshot_image = Image.frombytes(
                'RGB',
                (screenshot.width, screenshot.height),
                screenshot.rgb,
            )
            output, _ = real_time_object_detection(
                OUTPUT_WINDOW_NAME,
                output_shape,
                get_detections,
                screenshot_image,
                category_index,
                eval_config=detection_configs['eval_config'],
            )
            if output is None:
                break
