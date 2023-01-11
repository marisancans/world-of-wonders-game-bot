import cv2
from typing import List
import numpy as np
import math
from tesserocr import PyTessBaseAPI, PSM
import tesserocr
from pathlib import Path
import easyocr
from PIL import Image

from structures import Word, Letter, Option
import helper
import screenshot


def get_screenshot(screen: screenshot.ScreenShot):
    screenshot = screen.get_screen()

    return screenshot

def get_img_regions(screenshot):

    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
    h, w, _ = screenshot.shape
    grid_img = screenshot[int(h*0.1):int(h*0.6), :]

    # helper.show("img", grid_img, 1)
    gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # helper.show("thresh", thresh, 1)

    circle_img = screenshot[int(h * 0.55):, :]

    rows = circle_img.shape[0]

    gray_circle_img = cv2.cvtColor(circle_img, cv2.COLOR_BGR2GRAY)
    ret, thresh_circle_img = cv2.threshold(gray_circle_img, 220, 255, cv2.THRESH_BINARY)
    thresh_circle_img = cv2.erode(thresh_circle_img, np.ones((5, 5), np.uint8), iterations=1)
    thresh_circle_img = cv2.dilate(thresh_circle_img, np.ones((5, 5), np.uint8), iterations=4)

    # thresh_circle_img[(thresh_circle_img!=255)] = [0, 0, 0]

    # helper.show("thresh_circle_img", thresh_circle_img, 0)
    contours, hierarchy = cv2.findContours(thresh_circle_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for i, cnt in enumerate(contours):
        print(i)
        # cv2.polylines(circle_img, [cnt], 0, (0, 0, 255), 3)

        cx, cy, cw, ch = cv2.boundingRect(cnt)
        center = (cx + cw // 2, cy + ch // 2)
        radius = max(cw // w, ch // 2)

        h, w, _ = circle_img.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        cv2.circle(mask, center, radius - 10, (255, 255, 255), -1, cv2.LINE_AA)

        circle_img[(mask==0)] = [255, 255, 255]
        cv2.circle(circle_img, center, int(h*0.08), (255, 255, 255), -1, cv2.LINE_AA)

    # helper.show("circle_img", circle_img, 0)

    return grid_img, thresh, circle_img


def get_words(grid_img, thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    letters: List[Letter] = []

    for idx, contour in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        if area < 300:
            continue

        letter = Letter(idx, "", x, y, w, h)
        letters.append(letter)

        cv2.rectangle(grid_img, (x,y), (x+w,y+h), (0, 0, 255), 4)


    for letter in letters:
        cx = letter.x + letter.w / 2
        cy = letter.y + letter.h / 2

        cv2.circle(grid_img, (int(cx), int(cy)), 10, (0, 0, 255), -1)

    avg_w = sum(letter.w for letter in letters) / len(letters)
    avg_h = sum(letter.h for letter in letters) / len(letters)

    xs = np.array([letter.cx for letter in letters])
    ys = np.array([letter.cy for letter in letters])

    words: List[Word] = []

    for x in xs:
        matches = []

        for letter in letters:
            xdiff = abs(letter.cx - x)
            
            if xdiff < (avg_w * 0.2):
                matches.append(letter)

        if len(matches) <= 1:
            continue

        words.append(Word(matches))


    for y in ys:
        matches = []

        for letter in letters:
            ydiff = abs(letter.cy - y)
            
            if ydiff < (avg_h * 0.2):
                matches.append(letter)

        if len(matches) <= 1:
            continue

        words.append(Word(matches))

    clean = []

    for word in words:
        good = True

        for c in clean:
            if word.letters == c.letters:
                good = False
                break

        for previous, current in zip(word.letters, word.letters[1:]):
            dist = math.dist([previous.cx, previous.cy], [current.cx, current.cy])

            if dist > (max(avg_h, avg_w) * 1.5):
                good = False
                break


        if good:
            clean.append(word)

    return clean

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

    screen = screenshot.ScreenShot()
    

    for times in range(10):
        # base_img = cv2.imread("Screenshot_20230109-151355.png")
        base_img = screen.get_screen()
        
        grid_img, thresh, circle_img = get_img_regions(base_img)

        gray_circle_img = cv2.cvtColor(circle_img, cv2.COLOR_BGR2GRAY)
        # helper.show("gray_circle_img", gray_circle_img, 0)
        ret, circle_img_thresh = cv2.threshold(gray_circle_img, 200, 255, cv2.THRESH_BINARY)

        circle_img_thresh = cv2.bitwise_not(circle_img_thresh)

        # helper.show("circle_img_thresh", circle_img_thresh)
        contours, hierarchy = cv2.findContours(circle_img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # helper.show("circle_img", circle_img)

        # N O W
        options: List[Option] = []

        for cnt in contours:
            cv2.polylines(circle_img, [cnt], 0, (0, 0, 255), 1)

            x,y,w,h = cv2.boundingRect(cnt)
            p = 20

            char_img = gray_circle_img[y-p:y+h+p, x-p:x+w+p]

            img = Image.fromarray(char_img)

            with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_CHAR) as api:
                api.SetImage(img)    
                a = api.GetUTF8Text().strip().lower()
                options.append(Option(a, x, y, w, h))

                # cv2.putText(circle_img, a, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 155, 0), 2)

                if len(a) > 1:
                    print(a)
                    print("Why text is > 1 ?")
                    # helper.show("circle_img", circle_img)
                    helper.show("char_img", char_img)
                    exit(1)

                if a.isnumeric():
                    print("Numeric?")
                    exit(1)

        print("options:", *options)

        words = get_words(grid_img, thresh)

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





