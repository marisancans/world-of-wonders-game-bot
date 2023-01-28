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
    path.mkdir(parents=True, exist_ok=True)

    dataset = []

    for char_folder in path.iterdir():
        for char_path in char_folder.iterdir():
            img = cv2.imread(str(char_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       
            dataset.append([img, char_folder.name])

    return dataset



def save_to_dataset(dataset, crop_img, text, suffix):
    out_folder = Path(config.DATASET_DIRTY_PATH, suffix, text)
    out_folder.mkdir(parents=True, exist_ok=True)
    out_img = out_folder / f"{uuid.uuid4()}.png"

    # check if such a letter already exists:

    for other_char_img, other_char_text in dataset:
        if other_char_img.shape != crop_img.shape:
            continue
    
        # Due to image compression they can differ, so check structural similarity instead
        other_gray = cv2.cvtColor(other_char_img, cv2.COLOR_RGB2GRAY)
        crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

        h, w = crop_gray.shape
        if h < 7 or w < 7:
            continue
        
        score = compare_ssim(crop_gray, other_gray)

        if score > 0.95:
            return 

    cv2.imwrite(str(out_img), crop_img)


def preprocess_img(letter_crop):
    letter_crop = helper.resize_with_pad(letter_crop, (128, 128))
    img_t = torch.from_numpy(letter_crop).to(torch.float32) / 255.0
    img_t = img_t.permute(2, 0, 1)
    img_t = img_t.unsqueeze(0).cuda()
    return img_t


def guess_letters(words: List[Word], grid_img, grid_img_canvas, models):
    model, alphabet = models["cell"]

    # dataset = load_dataset("cells")

    for word in words:
        
        for letter in word.letters:
            
            # Skip already guessed letters
            # if letter.char != "":
            #     if letter.char != "*":
            #         continue

            letter_crop = grid_img[letter.y:letter.y + letter.h, letter.x:letter.x + letter.w]

            img_t = preprocess_img(letter_crop)

            logits =  model.forward(img_t)
            preds = torch.argmax(logits, dim=1)

            char = alphabet[preds[0]]

            # print(char)
            # helper.show("letter_crop", letter_crop)

            # if config.SAVE_TO_DATASET:
                # save_to_dataset(dataset, letter.crop, char, "cells")

            if char == "empty":
                char = "*"

            if char == "butterfly":
                char = "*"

            if char == "pine":
                char = "*"

            letter.char = char    

    for word in words:
        for letter in word.letters:
            cv2.putText(grid_img_canvas, letter.char.upper(), (int(letter.cx), int(letter.cy)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 127, 0), 5)
            helper.show("grid_img", grid_img_canvas, 1 )


def guess_circle_letters(circle_img, models):
    model, alphabet = models["circle"]


    ch, cw, _ = circle_img.shape
    center = (cw // 2, ch // 2)
    radius = int(cw * 0.31)

    h, w, _ = circle_img.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.circle(mask, center, radius - 10, (255, 255, 255), -1, cv2.LINE_AA)

    circle_img[(mask==0)] = [255, 255, 255]
    cv2.circle(circle_img, center, int(h*0.08), (255, 255, 255), -1, cv2.LINE_AA)

    gray_circle_img = cv2.cvtColor(circle_img, cv2.COLOR_RGB2GRAY)
    # helper.show("gray_circle_img", gray_circle_img, 1)

    ret, circle_img_thresh = cv2.threshold(gray_circle_img, 50, 255, cv2.THRESH_BINARY)
    circle_img_thresh = cv2.bitwise_not(circle_img_thresh)

    helper.show("circle_img_thresh", circle_img_thresh, 1)

    contours, hierarchy = cv2.findContours(circle_img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    options: List[Letter] = []

    dataset = load_dataset("circle")

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 5 or h < 20:
            continue

        letter_crop = circle_img[y:y+h, x:x+w]

        img_t = preprocess_img(letter_crop)

        logits =  model.forward(img_t)
        preds = torch.argmax(logits, dim=1)

        char = alphabet[preds[0]]
        print(char)
        
        if config.SAVE_TO_DATASET:
            save_to_dataset(dataset, letter_crop, char, "circle")
        
        # helper.show("letter_crop", letter_crop)

        option = Letter(len(options), char, x, y, w, h)
        options.append(option)

        cv2.putText(circle_img, char.upper(), (int(x + w), int(y + h)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 127, 0), 10)

    helper.show("circle_img", circle_img, 1)

        
    print("options:", *options)
    return options
