import cv2
from pathlib import Path
import json

def fs_json_load(path: Path):
    data = {}

    if not path.exists():
        raise Exception(f"{path} doesn't exist in json_load()")

    with open(str(path)) as f:
        data = json.load(f)
    return data


def show(name, img, waitKey=0):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(waitKey)
