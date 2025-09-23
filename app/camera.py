import cv2
import threading

class Camera:
    """A class to manage the camera thread and frame capture."""

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.video = None
        self.outputFrame = None
        self.lock = threading.Lock()
        self.thread = None
        self.is_running = False

    def start(self):
        if self.is_running:
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        self.video = cv2.VideoCapture(self.camera_index)
        if not self.video.isOpened():
            print(f"Error: Could not open camera at index {self.camera_index}.")
            self.is_running = False
            return

        while self.is_running:
            ret, frame = self.video.read()
            if not ret:
                print("Error: Can't receive frame. Exiting capture thread.")
                break

            with self.lock:
                self.outputFrame = frame.copy()

        self.video.release()
        self.is_running = False

    def get_jpeg_frame(self):
        while self.outputFrame is None:
            if not self.is_running and self.thread is not None:
                return None

        with self.lock:
            if self.outputFrame is None:
                return None

            (flag, encodedImage) = cv2.imencode(".jpg", self.outputFrame)
            if not flag:
                return None

        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"

    def stop(self):
        self.is_running = False
        if self.thread is not None:
            self.thread.join()
