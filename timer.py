import time

class Timer:
    def __init__(self):
        self.start_sec = None

    def start(self):
        self.start_sec = time.time()

    def stop(self):
        return time.time() - self.start_sec
