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

def get_img_regions(screenshot):
    h, w, _ = screenshot.shape
    grid_img = screenshot[int(h*0.1):int(h*0.5), :]

    circle_img = screenshot[int(h * 0.57):, :]

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


def get_possible_matches(possible_words, options):
    possible_words = [x for x in possible_words if len(x) <= len(options)]

    filtered = []
    chars = [o.char for o in options]

    for w in possible_words:
        if set(w) == set(chars):
            filtered.append(w)

    return filtered


def char_to_option(char, options):
    for option in options:
        if char != option.letter:
            continue

        return option


def swipe_guess(guess, options: List[Letter], circle_img):    
    print("Guessing:", guess)

    for previous, current in zip(guess, guess[1:]):
        prev_option = char_to_option(previous, options)
        current_option = char_to_option(current, options)
        cv2.line(circle_img, 
            (prev_option.x + prev_option.w // 2, prev_option.y + prev_option.h // 2), 
            (current_option.x + current_option.w // 2, current_option.y + current_option.h // 2), (255, 0, 0), 2)

    # helper.show("circle_img", circle_img)



def main():
    possible_words = helper.fs_json_load(Path("words_dictionary.json"))
    possible_words = list(possible_words.keys())

    client = phone_client.Client()

    for times in range(10):
        # base_img = cv2.imread("Screenshot_20230109-151355.png")

        base_img = None

        while not isinstance(base_img, np.ndarray):
            time.sleep(1)

            base_img = client.get_frame()
        
        grid_img, circle_img = get_img_regions(base_img)

        words = grid.get_words(grid_img)
        ocr.guess_letters(words, grid_img)
        options = ocr.guess_circle_letters(circle_img)


        for word in words:
            x1, y1, x2, y2 = word.bbox
            o = 15

            print(len(word.letters))
            cv2.rectangle(grid_img, (int(x1 + o), int(y1 + o)), (int(x2 - o), int(y2 - o)), (255, 0, 0), 5)

        matches = get_possible_matches(possible_words, options)

        # for match in matches:
            # swipe_guess(match, options, circle_img.copy())

        # helper.show("circle_img", circle_img, 1)
        # helper.show("grid_img", grid_img, 0)

        x = 0

    
if __name__ == "__main__":
    main()





