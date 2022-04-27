import os
import time
from PIL import Image
from mss import mss


SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
DT = 0.5
NUM_IMAGES = 1000
OUTDIR = "E:/Projects/fps/workspace/images/"


input_window = dict(
    left=SCREEN_WIDTH // 4 - 200,
    top=SCREEN_HEIGHT // 4 - 113,
    width=400,
    height=225,
)
with mss() as sct:
    i = 0
    for i in range(NUM_IMAGES):
        time.sleep(DT)
        screenshot = sct.grab(input_window)
        image = Image.frombytes(
            'RGB',
            (screenshot.width, screenshot.height),
            screenshot.rgb,
        )
        print(f"{i}/{NUM_IMAGES}", end="\r")
        image.save(os.path.join(OUTDIR, f"screenshot_{i:03d}.png"))