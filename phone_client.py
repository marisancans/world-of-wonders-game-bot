
import scrcpy
import helper
from adbutils import adb
from threading import Lock
import time
import cv2
import config
import math
import numpy as np

from sympy.abc import t
from sympy import Ray, Circle, Segment

def char_to_option(char, options):
    for option in options:
        if char != option.char:
            continue
        
        if option.used:
            continue

        return option

    return None

def directed_segment(line, length):
    p3 = Ray(*line.args).intersection(Circle(line.p1, length))[0]
    return line.func(line.p1, p3)


def extend_segment(x_from, y_from, x_to, y_to, extension):
    lenAB = math.sqrt(math.pow(x_from - x_to, 2.0) + math.pow(y_from - y_to, 2.0))

    x_to =int (x_to + (x_to - x_from) / lenAB * extension)
    y_to = int(y_to + (y_to - y_from) / lenAB * extension)

    return x_to, y_to

class Client():
    def __init__(self):
        # You can also pass an ADBClient instance to it
        adb.connect("127.0.0.1:5555")
        self.client = scrcpy.Client(device=adb.device_list()[0])
        self.last_frame = None
        self.work = True

        self.lock = Lock()

        def on_frame(frame):
            # If you set non-blocking (default) in constructor, the frame event receiver 
            # may receive None to avoid blocking event.
            if frame is not None:
                # frame is an bgr numpy ndarray (cv2' default format)
                # helper.show("viz", frame, 10)

                with self.lock:
                    self.last_frame = frame

        self.client.add_listener(scrcpy.EVENT_FRAME, on_frame)
        self.client.start(threaded=True)


    def get_frame(self):
        with self.lock:
            return self.last_frame


    def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        steps: int = 50,
        move_steps_delay: float = 0.005,
    ) -> None:
        """
        Swipe on screen

        Args:
            start_x: start horizontal position
            start_y: start vertical position
            end_x: start horizontal position
            end_y: end vertical position
            move_step_length: length per step
            move_steps_delay: sleep seconds after each step
        :return:
        """

        xs = np.linspace(start_x, end_x, steps)
        ys = np.linspace(start_y, end_y, steps)

        for x, y in zip(xs, ys):
            self.client.control.touch(x, y, scrcpy.ACTION_MOVE)

            time.sleep(move_steps_delay)



    def swipe_guess(self, guess, options, base_img):    
        print("Guessing:", guess)

        h, w, _ = base_img.shape

    
        offset = int(h * (1 - config.CIRCLE_OFFSET_DOWN))


        debug = base_img.copy()

        x_circle = w // 2
        y_circle = int(h * 0.735)
        

        cv2.circle(base_img, (x_circle, y_circle), 10, (200, 0, 0), 4)

        op_letters = []
        for g in guess:
            l = char_to_option(g, options)
            l.used = True
            op_letters.append(l)


        it = list(enumerate(zip(op_letters, op_letters[1:])))

        x_end = 0
        y_end = 0

        for idx, (prev_option, current_option) in it:
            
            cv2.line(debug, (0, offset), (w, offset), (255, 0, 255), 5)

            prev_option.used = True

            x_from = int(prev_option.x + (prev_option.w // 2))
            y_from = int(prev_option.y - (prev_option.h // 2) + offset)

            x_to = int(current_option.x + (current_option.w // 2))
            y_to = int(current_option.y - (current_option.h // 2) + offset)

            cv2.circle(debug, (x_from, y_from), 10, (0, 0, 255), 5)
            cv2.circle(debug, (x_to, y_to), 10, (0, 0, 255), 5)

            x_circle_extended, y_circle_extended = extend_segment(x_from, y_from, x_circle, y_circle, 50)
            x_to, y_to = extend_segment(x_circle_extended, y_circle_extended, x_to, y_to, 50)

            x_end = x_to
            y_end = y_to

            
           
            cv2.line(debug, (x_from, y_from), (x_circle_extended, y_circle_extended), (255, 0, 0), 5)
            helper.show("swiper", debug, 1)

            if idx == 0:
                self.client.control.touch(x_from, y_from, scrcpy.ACTION_DOWN)
            
        
            self.swipe(x_from, y_from, x_circle_extended, y_circle_extended, move_steps_delay=0.01, steps=25)
            self.swipe(x_circle_extended, y_circle_extended, x_to, y_to, move_steps_delay=0.01, steps=25)
           

        self.client.control.touch(x_end, y_end, scrcpy.ACTION_UP)
        helper.show("swiper", debug, 1)

        for o in options:
            o.used = False

