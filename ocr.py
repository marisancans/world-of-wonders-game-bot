import helper
from structures import Word, Letter
from PIL import Image
from typing import List

import math
import cv2
import numpy as np
import time
from pathlib import Path
import config
import uuid
from skimage.metrics import structural_similarity as compare_ssim
import torch

def load_dataset(suffx):
    here = Path(__file__).parent
    path = Path(here, config.DATASET_DIRTY_PATH, suffx)

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

    out_folder = Path(config.DATASET_DIRTY_PATH, suffix, text)
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


def preprocess_img(letter_crop):
    letter_crop = helper.resize_with_pad(letter_crop, (128, 128))
    img_t = torch.from_numpy(letter_crop).to(torch.float32) / 255.0
    img_t = img_t.permute(2, 0, 1)
    img_t = img_t.unsqueeze(0).cuda()
    return img_t

def guess_letters(words: List[Word], grid_img, models):
    model, alphabet = models["cell"]

    for word in words:
        for letter in word.letters:
            letter_crop = letter.crop.copy()
            img_t = preprocess_img(letter_crop)

            logits =  model.forward(img_t)
            preds = torch.argmax(logits, dim=1)

            char = alphabet[preds[0]]

            
            
            # print(char)
            # helper.show("letter_crop", letter_crop)

            if config.SAVE_TO_DATASET:
                save_to_dataset(letter.crop, char, "cells")

            if char == "empty":
                char = "*"
            letter.char = char           


def guess_circle_letters(circle_img, models):
    model, alphabet = models["circle"]

    gray_circle_img = cv2.cvtColor(circle_img, cv2.COLOR_RGB2GRAY)
    # helper.show("gray_circle_img", gray_circle_img, 0)
    ret, circle_img_thresh = cv2.threshold(gray_circle_img, 100, 255, cv2.THRESH_BINARY)
    circle_img_thresh = cv2.erode(circle_img_thresh, np.ones((5, 5), np.uint8), iterations=2)
    circle_img_thresh = cv2.dilate(circle_img_thresh, np.ones((5, 5), np.uint8), iterations=2)

    circle_img_thresh = cv2.bitwise_not(circle_img_thresh)

    helper.show("circle_img_thresh", circle_img_thresh, 1)
    contours, hierarchy = cv2.findContours(circle_img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    options: List[Letter] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 20 or h < 20:
            continue

        letter_crop = circle_img[y:y+h, x:x+w]

        img_t = preprocess_img(letter_crop)

        logits =  model.forward(img_t)
        preds = torch.argmax(logits, dim=1)

        char = alphabet[preds[0]]
        print(char)
        
        if config.SAVE_TO_DATASET:
            save_to_dataset(letter_crop, char, "circle")
        
        # helper.show("letter_crop", letter_crop)

        option = Letter(len(options), char, x, y, w, h, letter_crop)
        options.append(option)

        
    print("options:", *options)
    return options
