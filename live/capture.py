import cv2
import mss
import numpy as np


class ScreenCapture:
    def __init__(self, monitor_index=1):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_index]

    def grab(self):
        shot = self.sct.grab(self.monitor)
        bgra = np.asarray(shot)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)

    def close(self):
        self.sct.close()
