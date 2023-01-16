import scrcpy
import helper
from adbutils import adb
from threading import Lock

class Client():
    def __init__(self):

        # If you already know the device serial
        client = scrcpy.Client(device="DEVICE SERIAL")
        # You can also pass an ADBClient instance to it
        adb.connect("127.0.0.1:5555")
        client = scrcpy.Client(device=adb.device_list()[0])
        self.last_frame = None

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


