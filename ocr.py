import helper
from structures import Word, Letter
from PIL import Image
from typing import List

import math
import cv2
import tesserocr
import pytesseract
import numpy as np
import time
from pathlib import Path
import config
import uuid
from skimage.metrics import structural_similarity as compare_ssim


def load_dataset(suffx):
    here = Path(__file__).parent
    path = Path(here, config.DATASET_PATH, suffx)

    dataset = []

    for char_folder in path.iterdir():
        for char_path in char_folder.iterdir():
            img = cv2.imread(str(char_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       
            dataset.append([img, char_folder.name])

    return dataset



def save_to_dataset(crop_img, text, suffix):

    if text.isnumeric():
        text = "unknown"
    if len(text) != 1:
        text = "unknown"
    if not text.isalpha():
        text = "unknown"

    out_folder = Path(config.DATASET_PATH, suffix, text)
    out_folder.mkdir(parents=True, exist_ok=True)
    out_img = out_folder / f"{uuid.uuid4()}.png"

    # check if such a letter already exists:
    dataset = load_dataset(suffix)

    for other_char_img, other_char_text in dataset:
        if other_char_img.shape != crop_img.shape:
            continue
    
        # Due to image compression they can differ, so check structural similarity instead
        other_gray = cv2.cvtColor(other_char_img, cv2.COLOR_RGB2GRAY)
        crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
        
        score = compare_ssim(crop_gray, other_gray)
        print(score)

        if score > 0.95:
            print("The same image detected")
            # diff = cv2.absdiff(crop_img, other_char_img)
            # helper.show("diff", diff)
            return 

    cv2.imwrite(str(out_img), crop_img)

def guess_letters(words: List[Word], grid_img):
    for word in words:
        for letter in word.letters:
            letter_crop = letter.crop.copy()
            letter_crop = cv2.cvtColor(letter_crop, cv2.COLOR_RGB2GRAY)
            ret, letter_crop = cv2.threshold(letter_crop, 200, 255, cv2.THRESH_BINARY)

            # letter_crop = cv2.resize(letter_crop, (50, 50))
            text = pytesseract.image_to_string(letter_crop, lang='eng', config="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz --psm 10")
            text = text.strip().lower()
            print(text, np.count_nonzero(letter_crop) / (letter_crop.shape[0] * letter_crop.shape[1]))
            helper.show("letter_crop", letter_crop)

            if config.SAVE_TO_DATASET:
                save_to_dataset(letter.crop, text, "cells")

            letter.char = text           


def guess_circle_letters(circle_img):
    gray_circle_img = cv2.cvtColor(circle_img, cv2.COLOR_RGB2GRAY)
    # helper.show("gray_circle_img", gray_circle_img, 1)
    ret, circle_img_thresh = cv2.threshold(gray_circle_img, 100, 255, cv2.THRESH_BINARY)

    circle_img_thresh = cv2.bitwise_not(circle_img_thresh)

    helper.show("circle_img_thresh", circle_img_thresh, 1)
    contours, hierarchy = cv2.findContours(circle_img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    options: List[Letter] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        char_img_thresh = circle_img_thresh[y:y+h, x:x+w]
        char_img = circle_img[y:y+h, x:x+w]

        scale_percent = 50 # percent of original size
        width = int(char_img_thresh.shape[1] * scale_percent / 100)
        height = int(char_img_thresh.shape[0] * scale_percent / 100)
        dim = (width, height)

        char_img_thresh = cv2.resize(char_img_thresh, dim)
        p = 20
        char_img_thresh = cv2.copyMakeBorder(char_img_thresh, p, p, p, p, cv2.BORDER_CONSTANT)
        # char_img = cv2.resize(char_img, (50, 50))

        text = pytesseract.image_to_string(char_img_thresh, lang='eng', config="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz --psm 10")
        text = text.strip().lower()
        print(text)
        if config.SAVE_TO_DATASET:
            save_to_dataset(char_img, text, "circle")
        
        # helper.show("char_img", char_img_thresh)

        option = Letter(len(options), text, x, y, w, h, char_img)
        options.append(option)

        
    print("options:", *options)
    return options
