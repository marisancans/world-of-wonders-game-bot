import cv2
from typing import List
import numpy as np
from structures import Letter, Word
import helper
import config
import torch


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


def preprocess_img(grid_img):
    grid_img = cv2.resize(grid_img, (256, 256))
    img_t = torch.from_numpy(grid_img).to(torch.float32) / 255.0
    img_t = img_t.permute(2, 0, 1)
    img_t = img_t.unsqueeze(0).to(config.DEVICE)
    return img_t


def get_words(grid_img, grid_img_canvas, models):
    model = models["grid"]


    image = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = cv2.resize(image, (256, 256))
    image = np.moveaxis(image, -1, 0)
    image = torch.from_numpy(image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        model.eval()
        logits = model(image)
    pr_masks = logits.sigmoid()

    pred_mask = pr_masks.cpu().detach().numpy().squeeze()
    pred_mask = (pred_mask * 255).astype(np.uint8)

    helper.show("pred_mask", pred_mask, 1)

    h, w, c = grid_img.shape
    pred_mask = cv2.resize(pred_mask, (w, h))

    grid_img_masked = cv2.bitwise_and(grid_img, grid_img, mask = pred_mask)
    helper.show("grid_img_masked", grid_img_masked, 1)

    grid_thersh = cv2.cvtColor(grid_img_masked, cv2.COLOR_BGR2GRAY)
    helper.show("grid_thersh", grid_thersh, 1)

    ret, pred_mask = cv2.threshold(grid_thersh, 90, 255, cv2.THRESH_BINARY)

    helper.show("grid_thersh", grid_thersh, 1)


    fin_grid = cv2.bitwise_and(grid_img, grid_img, mask = pred_mask)
    helper.show("fin_grid", fin_grid, 1)
    fin_grid = cv2.cvtColor(fin_grid, cv2.COLOR_BGR2GRAY)
    ret, fin_grid_thresh = cv2.threshold(fin_grid, 0, 255, cv2.THRESH_BINARY)

    fin_grid_thresh = cv2.dilate(fin_grid_thresh, np.ones((2, 2), np.uint8), iterations=1)
    # fin_grid_thresh = cv2.erode(fin_grid_thresh, np.ones((5, 5), np.uint8), iterations=1)
    

    helper.show("fin_grid_thresh", fin_grid_thresh, 1)

    contours, _ = cv2.findContours(fin_grid_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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