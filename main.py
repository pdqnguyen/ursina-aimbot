import numpy as np
from PIL import Image
from mss import mss
from detection import (load_model, get_model_detection_function, load_labels,
                       create_output_window, real_time_object_detection)
import pyautogui


SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
GAME_X, GAME_Y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
WINDOW_X, WINDOW_Y = 400, 225
OUTPUT_WINDOW_NAME = 'Aimbot test'

MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
MODEL_CKPT = 'ckpt-9'
MODEL_VERSION = 'v3'
MODELS_DIR = 'E:/Projects/fps/workspace/models/'


def get_box_pix(b, window_width, window_height):
    y0, x0, y1, x1 = b
    left, right, top, bottom = (x0 * window_width, x1 * window_width, y0 * window_height, y1 * window_height)
    return left, right, top, bottom


def get_box_dist(b, mouse, window):
    left, right, top, bottom = get_box_pix(b, window['width'], window['height'])
    width = abs(left - top) / 2
    height = abs(top - bottom) / 2
    xmid = 0.5 * (left + right) + window['left']
    ymid = 0.5 * (top + bottom) + window['top']
    dx = xmid - mouse[0]
    dy = ymid - mouse[1]
    on_target = (abs(dx) < width) and (abs(dy) < height)
    return dx, dy, on_target


if __name__ == "__main__":
    input_window = dict(
        left=(GAME_X - WINDOW_X) // 2,
        top=(GAME_Y - WINDOW_Y) // 2,
        width=WINDOW_X,
        height=WINDOW_Y,
    )
    output_pos = (SCREEN_WIDTH - WINDOW_X, SCREEN_HEIGHT - WINDOW_Y - 30)
    output_shape = (WINDOW_X, WINDOW_Y)

    detection_model, detection_configs = load_model(
        model_name=MODEL_NAME,
        model_ckpt=MODEL_CKPT,
        model_version=MODEL_VERSION,
        models_dir=MODELS_DIR
    )
    get_detections = get_model_detection_function(detection_model)

    # Load label map (for plotting)
    category_index = load_labels(detection_configs['eval_input_config'])

    # Visualization
    create_output_window(OUTPUT_WINDOW_NAME, output_pos)

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
            boxes = np.array(output['detection_boxes'])[0, :]
            scores = np.array(output['detection_scores'])[0, :]
            positives = (scores > 0.5)
            mouse_x, mouse_y = pyautogui.position()
            nearest_dist_x = np.inf
            nearest_dist_y = np.inf
            nearest_on_target = False
            for box, score in zip(boxes[positives], scores[positives]):
                box_dx, box_dy, box_on_target = get_box_dist(box, (mouse_x, mouse_y), input_window)
                if box_dx**2 + box_dy**2 < nearest_dist_x**2 + nearest_dist_y**2:
                    nearest_dist_x = box_dx
                    nearest_dist_y = box_dy
                    nearest_on_target = box_on_target
            if nearest_dist_x < np.inf:
                if nearest_on_target:
                    pyautogui.mouseDown()
                else:
                    pyautogui.mouseUp()
                mouse_x += nearest_dist_x
                mouse_y += nearest_dist_y
                pyautogui.moveTo(mouse_x, mouse_y)
            else:
                pyautogui.mouseUp()
