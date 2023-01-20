import cv2
from typing import List
import numpy as np
from pathlib import Path

import time
from structures import Word, Letter
import helper
import phone_client
import grid
import ocr
import re
import itertools
import config
import uuid
from character_classification import LitAlphabet, AlphaDataset
from pytorch_lightning import Trainer
from words import get_possible_matches




def get_img_regions(screenshot):
    h, w, _ = screenshot.shape
    grid_img = screenshot[int(h*0.1):int(h*0.55), :]
    # helper.show("grid_img", grid_img)

    circle_img = screenshot[int(h * 0.57):, :]

    if config.SAVE_TO_DATASET:
        out = Path("dataset", "grid")
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{uuid.uuid4()}.png"), grid_img)

    gray_circle_img = cv2.cvtColor(circle_img, cv2.COLOR_RGB2GRAY)
    ret, thresh_circle_img = cv2.threshold(gray_circle_img, 220, 255, cv2.THRESH_BINARY)
    thresh_circle_img = cv2.erode(thresh_circle_img, np.ones((5, 5), np.uint8), iterations=1)
    thresh_circle_img = cv2.dilate(thresh_circle_img, np.ones((5, 5), np.uint8), iterations=4)

    # helper.show("thresh_circle_img", thresh_circle_img, 0)

    ch, cw, _ = circle_img.shape
    center = (cw // 2, ch // 2)
    radius = int(cw * 0.31)

    h, w, _ = circle_img.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.circle(mask, center, radius - 10, (255, 255, 255), -1, cv2.LINE_AA)

    circle_img[(mask==0)] = [255, 255, 255]
    cv2.circle(circle_img, center, int(h*0.08), (255, 255, 255), -1, cv2.LINE_AA)

    # helper.show("circle_img", circle_img, 0)

    return grid_img, circle_img



def get_model(name, version):
    ds_path = Path(f"./dataset_clean/{name}")
    model = LitAlphabet.load_from_checkpoint(
        f"logs_{name}/lightning_logs/version_{version}/checkpoints/best.ckpt",
        data_dir=ds_path
    )
    
    model = model.eval().cuda()

    alphabet = [x.name for x in sorted(ds_path.iterdir())]

    return model, alphabet




def main():
    possible_words = helper.fs_json_load(Path("words_dictionary.json"))
    possible_words = list(possible_words.keys())

    client = phone_client.Client()

    models = {
        "cell": get_model("cells", 0), 
        "cicle": get_model("circle", 0),
        "grid": get_grid_model()
    }


    for times in range(10):
        base_img = None

        while not isinstance(base_img, np.ndarray):
            time.sleep(1)

            base_img = client.get_frame()
        
        grid_img, circle_img = get_img_regions(base_img)
        grid_img_canvas = grid_img.copy()

        words = grid.get_words(grid_img, grid_img_canvas)
        ocr.guess_letters(words, grid_img, models)
        options = ocr.guess_circle_letters(circle_img, models)


        for word in words:
            x1, y1, x2, y2 = word.bbox
            o = 25

            cv2.rectangle(grid_img_canvas, (int(x1 + o), int(y1 + o)), (int(x2 - o), int(y2 - o)), (0, 0, 0), 8)

            for letter in word.letters:
                cv2.putText(grid_img_canvas, letter.char, (int(letter.x), int(letter.cy)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)


        for word in words:
            matches = get_possible_matches(possible_words, word)
            print(matches)

        # for match in matches:
            # swipe_guess(match, options, circle_img.copy())

        # helper.show("circle_img", circle_img, 1)
        helper.show("grid_img", grid_img_canvas, 0)

        x = 0


    
if __name__ == "__main__":
    main()





