import cv2
import numpy as np
from pathlib import Path

import time
import helper
import phone_client



import grid
import ocr

from character_classification import LitAlphabet
from grid_segmentation import GridModel
from words import get_possible_matches
import config
import uuid
import scrcpy


def get_img_regions(screenshot):
    h, w, _ = screenshot.shape
    grid_img = screenshot[int(h*config.CORCLE_OFFSET_UP):int(h*0.55), :]
   

    if config.SAVE_TO_DATASET:
        out = Path(config.DATASET_DIRTY_PATH, "grid")
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{uuid.uuid4()}.png"), grid_img)

    return grid_img


def get_circle_region(screenshot):
    h, w, _ = screenshot.shape

    circle_img = screenshot[int(h * config.CIRCLE_OFFSET_DOWN):, :].copy()
    # helper.show("circle_img", circle_img, 0)

    return circle_img

def get_alpha_model(name, version, ckpt_name):
    ds_path = Path(f"./dataset_clean/{name}")

    model = LitAlphabet.load_from_checkpoint(
        f"logs_{name}/lightning_logs/version_{version}/checkpoints/{ckpt_name}",
        data_dir=ds_path
    )
    
    model = model.eval().to(config.DEVICE)

    alphabet = [x.name for x in sorted(ds_path.iterdir())]

    print(f"Loaded {version} model: {ckpt_name}")

    return model, alphabet

def get_grid_model(name, version, ckpt_name):
    model = GridModel.load_from_checkpoint(
        f"logs_{name}/lightning_logs/version_{version}/checkpoints/{ckpt_name}"
    )

    print(f"Loaded grid model: {ckpt_name}")
    
    model = model.eval().to(config.DEVICE) 
    return model


# Check if victory or claim screen
def check_buttons(client):
    base_img = client.get_frame()
    helper.show("base_img", base_img, 1)

    def check_level():
        gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        congrats_crop = thresh[400:500, 250:850]
        # cv2.imwrite("congrats.png", congrats_crop)
        congrats_truth = cv2.imread("congrats.png", cv2.IMREAD_GRAYSCALE)

        diff = cv2.absdiff(congrats_crop, congrats_truth)
        helper.show("diff", diff, 1)
        avg = np.average(diff)

        if avg < 5:
            print("Clicking level button")
            client.client.control.touch(523, 1778, scrcpy.ACTION_DOWN)
            client.client.control.touch(523, 1778, scrcpy.ACTION_UP)
            time.sleep(10)
            return True
        

    def check_claim():
        gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        claim_crop = thresh[1835:1950, 372:744]
        helper.show("claim_crop", claim_crop, 1)
        # helper.show("thresh", thresh)
        # cv2.imwrite("claim.png", claim_crop)
        claim_truth = cv2.imread("claim.png", cv2.IMREAD_GRAYSCALE)

        diff = cv2.absdiff(claim_crop, claim_truth)
        helper.show("diff", diff, 1)
        avg = np.average(diff)

        if avg < 5:
            print("Clicking claim button")
            client.client.control.touch(544, 1913, scrcpy.ACTION_DOWN)
            client.client.control.touch(544, 1913, scrcpy.ACTION_UP)
            time.sleep(10)
            check_level()
            return True


    check_claim()
    check_level()
    return False



def type_words(models, client, words, options, possible_words):
    aviable_chars = [x.char for x in options]

    tried = []

    # Try larger words first 
    words = sorted(words, key=lambda x: len(x))
    
    for word in words:
        pattern = [x.char for x in word.letters]
        
        is_skip = all([x != "*" for x in pattern])

        t = f"Pattern: {pattern}"

        if is_skip:
            t += f"{t} skipping"

        print(t)

        if is_skip:
            continue
        
        print("Getting matches...")
        matches = get_possible_matches(possible_words, aviable_chars, pattern, words)

        if not matches:
            continue

        for idx, match in enumerate(matches):
            if match in tried:
                continue

            base_img = client.get_frame()
            grid_img = get_img_regions(base_img)
            grid_img_canvas = grid_img.copy()
        
            ocr.guess_letters(words, grid_img, grid_img_canvas, models)
            new_pattern = [x.char for x in word.letters]

            # Word state updated
            if new_pattern != pattern:
                print("Pattern updated")
                break

            print(idx, "/", len(matches), "Guessing:", match)
            client.swipe_guess(match, options, base_img)
            tried.append(match)


            if check_buttons(client):
                return
            
        print("All matches tried")
    print("All words tried")
    return


def main():
    cv2.startWindowThread()
    possible_words = helper.fs_json_load(Path("words_dictionary.json"))
    possible_words = list(possible_words.keys())

    client = phone_client.Client()

    models = {
        "cell": get_alpha_model("cells", 0, "best-v1.ckpt"), 
        "circle": get_alpha_model("circle", 0, "best-v1.ckpt"),
        "grid": get_grid_model("grid", 0, "best-v1.ckpt")
    }


    for times in range(1000):
        base_img = None

        while not isinstance(base_img, np.ndarray):
            print("Getting frame")
            time.sleep(1)

            base_img = client.get_frame()

        # Loop
        while True:   
            check_buttons(client)

            base_img = client.get_frame()
            grid_img = get_img_regions(base_img)
            circle_img = get_circle_region(base_img)
            grid_img_canvas = grid_img.copy()
        
            words = grid.get_words(grid_img, grid_img_canvas, models)
            ocr.guess_letters(words, grid_img, grid_img_canvas, models)
            options = ocr.guess_circle_letters(circle_img, models)

            type_words(models, client, words, options, possible_words)


                
    
if __name__ == "__main__":
    main()





