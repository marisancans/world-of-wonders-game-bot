import cv2
from typing import List
import numpy as np
from structures import Letter, Word
import helper
import config

def merge_values(values, threshold):
    found = False

    for i, c in enumerate(values):
        dist = np.abs(values - c)

        close_mask = dist < threshold
        close_mask[i] = False

        close = values[close_mask]

        if len(close) == 0:
            continue

        other = values[dist > threshold]
        
        close_avg = np.average(close)
        values = np.concatenate([np.array([close_avg]), other]) 
        found = True
        break

    if found:
        values = merge_values(values, threshold)

    return values

def find_letter_by_col_row(col_idx, row_idx, letters):
    for letter in letters:
        if letter.col != col_idx or letter.row != row_idx:
            continue

        return letter

    return None


def split_letters(letters, mode):
    stack: List[List[Letter]] = [[]]
    stack_id = len(stack) - 1

    if mode == "row":
        letters = sorted(letters, key=lambda x: x.row)
    else:
        letters = sorted(letters, key=lambda x: x.col)

    if mode == "row":
        last_idx = letters[0].row - 1
    else:
        last_idx = letters[0].col - 1


    for letter in letters:
        if mode == "row":
            if letter.row - 1 != last_idx:
                stack.append([])
                stack_id += 1
        else:
            if letter.col - 1!= last_idx:
                stack.append([])
                stack_id += 1

        stack[stack_id].append(letter)  

        if mode == "row":
            last_idx = letter.row
        else:
            last_idx = letter.col

    return stack


def stack_search(letters, idxs_a, idxs_b, mode, grid_img):
    words: List[Word] = []

    for idx_a in idxs_a:
        accumulate = []

        for idx_b in idxs_b:
            if mode == "row":
                found_letter = find_letter_by_col_row(idx_a, idx_b, letters)
            else:
                found_letter = find_letter_by_col_row(idx_b, idx_a, letters)

            if found_letter:
                accumulate.append(found_letter)
                # cv2.circle(grid_img, (int(found_letter.cx), int(found_letter.cy)), 10, (0, 0, 255), 5)
                # helper.show("grid_img", grid_img, 0)

        if len(accumulate) <= 1:
            continue

        # Col or row can contain multiple words
        stack = split_letters(accumulate, mode)
        
        for s in stack:
            if len(s) <= 1:
                continue

            words.append(Word(s))

    return words

def get_words(grid_img, grid_img_canvas):
    gray = cv2.cvtColor(grid_img, cv2.COLOR_RGB2GRAY)

    # ret, grid_img = cv2.threshold(grid_img, 220, 255, cv2.THRESH_BINARY)
    # helper.show("grid_img", grid_img)

    ret, thresh_empty_from = cv2.threshold(gray, 220, 255, cv2.THRESH_TRIANGLE)
    ret, thresh_empty_to = cv2.threshold(gray, 230, 255, cv2.THRESH_OTSU)

    ret, thresh_occupied_from = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
    ret, thresh_occupied_to = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    thresh_occupied = cv2.bitwise_and(thresh_occupied_from, thresh_occupied_to)
    thresh_occupied = cv2.bitwise_not(thresh_occupied)


    thresh_empty = cv2.bitwise_and(thresh_empty_from, thresh_empty_to)

    comb = cv2.bitwise_or(thresh_empty, thresh_occupied)
    
    helper.show("thresh_empty_from", thresh_empty_from, 1)
    helper.show("thresh_empty_to", thresh_empty_to, 1)
    helper.show("gray", gray, 1)
    helper.show("comb", comb, 1)
    helper.show("thresh_empty", thresh_empty, 1)
    helper.show("thresh_occupied", thresh_occupied, 0)

    contours, _ = cv2.findContours(comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    letters: List[Letter] = []

    areas = [cv2.contourArea(cnt) for cnt in contours]
    area_avg = np.array(areas).mean()

    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < 1000: # TODO: Should check standart deviation?
            continue
        
        letter_crop = grid_img[y:y + h, x:x + w]
        letter = Letter(idx, "", x, y, w, h, letter_crop)
        letters.append(letter)
        cv2.polylines(grid_img_canvas, [contour], True, (0, 255, 0), 5)

    if config.DEBUG:
        helper.show("grid_img", grid_img_canvas, 1)

    avg_w = sum(letter.w for letter in letters) / len(letters)
    avg_h = sum(letter.h for letter in letters) / len(letters)

    x_mean_all = [letter.cx for letter in letters]
    x_mean_all = np.array(x_mean_all)
    x_mean_all.sort()

    y_mean_all = [letter.cy for letter in letters]
    y_mean_all = np.array(y_mean_all)
    y_mean_all.sort()

    # combine close values
    x_mean = merge_values(x_mean_all, avg_w * 0.1)
    y_mean = merge_values(y_mean_all, avg_h * 0.1)

    x_mean = np.sort(x_mean)
    y_mean = np.sort(y_mean)

    # Assign col
    for col_idx, x in enumerate(x_mean):
        for letter in letters:
            xdiff = abs(letter.cx - x)

            if xdiff < (avg_w * 0.2):
                letter.col = col_idx

    # Assign row
    for row_idx, y in enumerate(y_mean):
        for letter in letters:
            ydiff = abs(letter.cy - y)
            
            if ydiff < (avg_h * 0.2):
                letter.row = row_idx

    # Create words from letter grid
    col_idxs = np.arange(len(x_mean))
    row_idxs = np.arange(len(y_mean))

    words_vertical = stack_search(letters, col_idxs, row_idxs, "row", grid_img)
    words_horizontal = stack_search(letters, row_idxs, col_idxs, "col", grid_img)

    words: List[Word] = words_vertical + words_horizontal    
 
    return words