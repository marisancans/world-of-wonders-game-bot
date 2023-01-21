
import scrcpy
import helper
from adbutils import adb
from threading import Lock
import time

class Client():
    def __init__(self):

        # If you already know the device serial
        client = scrcpy.Client(device="DEVICE SERIAL")
        # You can also pass an ADBClient instance to it
        adb.connect("127.0.0.1:5555")
        client = scrcpy.Client(device=adb.device_list()[0])
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

        client.add_listener(scrcpy.EVENT_FRAME, on_frame)
        client.start(threaded=True)

    def get_frame(self):
        with self.lock:
            return self.last_frame

# def char_to_option(char, options):
#     for option in options:
#         if char != option.letter:
#             continue

#         return option
    # def swipe_guess(guess, options: List[Letter], circle_img):    
    #     print("Guessing:", guess)

    #     for previous, current in zip(guess, guess[1:]):
    #         prev_option = char_to_option(previous, options)
    #         current_option = char_to_option(current, options)
    #         cv2.line(circle_img, 
    #             (prev_option.x + prev_option.w // 2, prev_option.y + prev_option.h // 2), 
    #             (current_option.x + current_option.w // 2, current_option.y + current_option.h // 2), (255, 0, 0), 2)

    #     # helper.show("circle_img", circle_img)

