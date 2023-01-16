import cv2
from typing import List
import numpy as np
import math
from tesserocr import PyTessBaseAPI, PSM
import tesserocr
from pathlib import Path
from PIL import Image
import time
from structures import Word, Letter, Option
import helper
import phone_client
import grid

def get_img_regions(screenshot):

    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
    h, w, _ = screenshot.shape
    grid_img = screenshot[int(h*0.1):int(h*0.5), :]

    circle_img = screenshot[int(h * 0.55):, :]

    rows = circle_img.shape[0]

    gray_circle_img = cv2.cvtColor(circle_img, cv2.COLOR_BGR2GRAY)
    ret, thresh_circle_img = cv2.threshold(gray_circle_img, 220, 255, cv2.THRESH_BINARY)
    thresh_circle_img = cv2.erode(thresh_circle_img, np.ones((5, 5), np.uint8), iterations=1)
    thresh_circle_img = cv2.dilate(thresh_circle_img, np.ones((5, 5), np.uint8), iterations=4)

    # thresh_circle_img[(thresh_circle_img!=255)] = [0, 0, 0]

    # helper.show("thresh_circle_img", thresh_circle_img, 0)
    contours, hierarchy = cv2.findContours(thresh_circle_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, reverse=True, key=lambda x: cv2.contourArea(x))
    cnt = contours[0]

    cv2.polylines(circle_img, [cnt], 0, (0, 0, 255), 3)

    cx, cy, cw, ch = cv2.boundingRect(cnt)
    center = (cx + cw // 2, cy + ch // 2)
    radius = max(cw // w, ch // 2)

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
    chars = [o.letter for o in options]

    for w in possible_words:
        if set(w) == set(chars):
            filtered.append(w)

    return filtered


def char_to_option(char, options):
    for option in options:
        if char != option.letter:
            continue

        return option


def swipe_guess(guess, options: List[Option], circle_img):    
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

        words = grid.get_cells(grid_img)

        gray_circle_img = cv2.cvtColor(circle_img, cv2.COLOR_BGR2GRAY)
        # helper.show("gray_circle_img", gray_circle_img, 0)
        ret, circle_img_thresh = cv2.threshold(gray_circle_img, 200, 255, cv2.THRESH_BINARY)

        circle_img_thresh = cv2.bitwise_not(circle_img_thresh)

        # helper.show("circle_img_thresh", circle_img_thresh)
        contours, hierarchy = cv2.findContours(circle_img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # helper.show("circle_img", circle_img)

        # N O W
        # options: List[Option] = []

        # for cnt in contours:
        #     cv2.polylines(circle_img, [cnt], 0, (0, 0, 255), 1)

        #     x,y,w,h = cv2.boundingRect(cnt)
        #     p = 20

        #     char_img = gray_circle_img[y-p:y+h+p, x-p:x+w+p]

        #     img = Image.fromarray(char_img)

        #     with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_CHAR) as api:
        #         api.SetImage(img)    
        #         a = api.GetUTF8Text().strip().lower()
        #         options.append(Option(a, x, y, w, h))

        #         # cv2.putText(circle_img, a, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 155, 0), 2)

        #         if len(a) > 1:
        #             print(a)
        #             print("Why text is > 1 ?")
        #             # helper.show("circle_img", circle_img)
        #             helper.show("char_img", char_img)
        #             exit(1)

        #         if a.isnumeric():
        #             print("Numeric?")
        #             exit(1)

        # print("options:", *options)

        

        for word in words:
            x1, y1, x2, y2 = word.bbox
            o = 15

            print(len(word.letters))
            cv2.rectangle(grid_img, (int(x1 + o), int(y1 + o)), (int(x2 - o), int(y2 - o)), (255, 0, 0), 5)

        matches = get_possible_matches(possible_words, options)

        for match in matches:
            swipe_guess(match, options, circle_img.copy())

        # helper.show("circle_img", circle_img, 1)
        # helper.show("grid_img", grid_img, 0)

        x = 0

    
if __name__ == "__main__":
    main()





